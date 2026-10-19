import os
import json
import torch
import swanlab
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm
from data_process import NERDataset
from model import BertForNER
from utils import set_seed, decode_predict, NEREvaluator
import argparse


class Trainer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.num_epochs = config["epochs"]
        self.best_f1 = 0.0
        self.model = None
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self.label2id = None
        self.id2label = None
        self.evaluator = NEREvaluator()

    def save_dir(self):
        dir_path = self.config.get("outdir")
        if not dir_path:
            dir_path = os.path.dirname(self.config["save_path"])
        return dir_path

    def get_path(self):
        return os.path.join(self.save_dir(), "best_model.pt")

    def load_data(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config["model_path"], use_fast=True)

        base_config = {
            'max_len': self.config["max_len"],
            'align_type': self.config.get("align_type", "ignore")
        }
        train_set = NERDataset({**base_config, 'data_path': os.path.join(self.config["data_dir"], self.config["train_file"])}, tokenizer)
        dev_set = NERDataset({**base_config, 'data_path': os.path.join(self.config["data_dir"], self.config["dev_file"]), 'label2id': train_set.label2id}, tokenizer)
        test_set = NERDataset({**base_config, 'data_path': os.path.join(self.config["data_dir"], self.config["test_file"]), 'label2id': train_set.label2id}, tokenizer)

        self.train_loader = train_set.get_loader(self.config["batch_size"], True)
        self.dev_loader = dev_set.get_loader(self.config["batch_size"], False)
        self.test_loader = test_set.get_loader(self.config["batch_size"], False)
        self.label2id = train_set.label2id
        self.id2label = train_set.id2label

    def init_model(self):
        self.config["num_labels"] = len(self.label2id)
        self.model = BertForNER(self.config).to(self.device)

    def save_model(self):
        dir_path = self.save_dir()
        os.makedirs(dir_path, exist_ok=True)
        name = "best_model"

        torch.save(self.model.state_dict(), self.get_path())
        with open(os.path.join(dir_path, f"{name}_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        with open(os.path.join(dir_path, f"{name}_label2id.json"), "w", encoding="utf-8") as f:
            json.dump(self.label2id, f, ensure_ascii=False, indent=2)
        AutoTokenizer.from_pretrained(self.config["model_path"], use_fast=True).save_pretrained(
            os.path.join(dir_path, f"{name}_tokenizer")
        )
       

    def train(self):
        swanlab.init(project="bert-ner1", config={
            "learning_rate": self.config["learning_rate"],
            "batch_size": self.config["batch_size"],
            "epochs": self.config["epochs"],
            "max_len": self.config["max_len"],
            "model_name": self.config["model_path"],
            "dropout_rate": self.config.get("dropout_rate", 0.1),
            "align_type": self.config.get("align_type", "ignore"),
            "num_labels": len(self.label2id)
        })

        optimizer = AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        weight_path = self.get_path()
        patience = self.config.get("patience", 3)
        counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss, count, total_valid_tokens = 0, 0, 0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                optimizer.zero_grad()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                mask = (attention_mask.reshape(-1) == 1) & (labels.reshape(-1) != -100)
                flat_logits = logits.reshape(-1, logits.size(-1))[mask]
                flat_labels = labels.reshape(-1)[mask]

                num_valid = flat_labels.numel()
                total_valid_tokens += num_valid
                if num_valid == 0:
                    continue

                loss_val = loss_fn(flat_logits, flat_labels)
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss_val.item()
                count += 1

            avg_train_loss = total_loss / max(count, 1)
            print(f"Epoch {epoch+1} 有效标签token数: {total_valid_tokens}")

            eval_loss, eval_f1, eval_precision, eval_recall, _, _ = self.eval()
            print(f"第 {epoch + 1} 轮 | 训练loss: {avg_train_loss:.4f} | 验证 P/R/F1: {eval_precision:.4f}/{eval_recall:.4f}/{eval_f1:.4f}")

            swanlab.log({
                "train/loss": avg_train_loss, "eval/loss": eval_loss,
                "eval/precision": eval_precision, "eval/recall": eval_recall, "eval/f1": eval_f1, "epoch": epoch + 1
            })

            if eval_f1 > self.best_f1 or (epoch == 0 and not os.path.exists(weight_path)):
                self.best_f1 = eval_f1
                self.save_model()
                print(f"保存最佳模型 | F1: {eval_f1:.4f}")
                swanlab.log({"best/f1": eval_f1, "best/epoch": epoch + 1})
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        print(f"最佳F1: {self.best_f1:.4f}")
        self.test()
        swanlab.finish()

    def eval(self):
        self.model.eval()
        total_loss, count = 0, 0
        all_preds, all_labels = [], []
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        with torch.no_grad():
            for batch in self.dev_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                mask = (attention_mask.reshape(-1) == 1) & (labels.reshape(-1) != -100)
                valid_logits = logits.reshape(-1, logits.size(-1))[mask]
                valid_labels = labels.reshape(-1)[mask]

                if valid_labels.numel() > 0:
                    total_loss += loss_fct(valid_logits, valid_labels).item()
                    count += 1

                preds = torch.argmax(logits, dim=-1)
                batch_preds, batch_labels = decode_predict(input_ids, preds, labels, attention_mask, self.id2label)
                all_preds.extend(batch_preds)
                all_labels.extend(batch_labels)

        avg_loss = total_loss / max(count, 1)
        f1, precision, recall = self.evaluator.evaluate(all_labels, all_preds)
        return avg_loss, f1, precision, recall, all_labels, all_preds

    def test(self):
        weight_path = self.get_path()
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=-1)
                batch_preds, batch_labels = decode_predict(input_ids, preds, labels, attention_mask, self.id2label)
                all_preds.extend(batch_preds)
                all_labels.extend(batch_labels)

        self.evaluator.print_report(all_labels, all_preds)
        f1, precision, recall = self.evaluator.evaluate(all_labels, all_preds)
        print(f" 测试集 P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")
        swanlab.log({"test/precision": precision, "test/recall": recall, "test/f1": f1})

    def run(self):
        self.load_data()
        self.init_model()
        self.train()


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("config_path")
    args = pa.parse_args()
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    set_seed(config.get("seed", 42))
    trainer = Trainer(config)
    trainer.run()
