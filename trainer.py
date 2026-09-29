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


    def load_data(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"],
            use_fast=True
        )

        train_config = {
            'data_path': os.path.join(self.config["data_dir"], self.config["train_file"]),
            'max_len': self.config["max_len"],
            'align_type': self.config.get("align_type", "ignore")
        }
        dev_config = {
            'data_path': os.path.join(self.config["data_dir"], self.config["dev_file"]),
            'max_len': self.config["max_len"],
            'align_type': self.config.get("align_type", "ignore")
        }
        test_config = {
            'data_path': os.path.join(self.config["data_dir"], self.config["test_file"]),
            'max_len': self.config["max_len"],
            'align_type': self.config.get("align_type", "ignore")
        }

        train_set = NERDataset(train_config, tokenizer)
        tmp = train_set.label2id
        dev_config['label2id'] = tmp
        test_config['label2id'] = tmp

        dev_set = NERDataset(dev_config, tokenizer)
        test_set = NERDataset(test_config, tokenizer)

        self.train_loader = train_set.get_loader(self.config["batch_size"], True)
        self.dev_loader = dev_set.get_loader(self.config["batch_size"], False)
        self.test_loader = test_set.get_loader(self.config["batch_size"], False)

        self.label2id = tmp
        self.id2label = train_set.id2label

    def init_model(self):
        self.config["num_labels"] = len(self.label2id)
        self.model = BertForNER(self.config)
        self.model.to(self.device)

    def save_model(self):
        save_dir = os.path.dirname(self.config["save_path"])
        os.makedirs(save_dir, exist_ok=True)

        name = os.path.basename(self.config["save_path"]).split('.')[0]

        torch.save(self.model.state_dict(), self.config["save_path"])

        with open(os.path.join(save_dir, name + "_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f)

        with open(os.path.join(save_dir, name + "_label2id.json"), "w", encoding="utf-8") as f:
            json.dump(self.label2id, f)

        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_path"],
            use_fast=True
        )
        tokenizer.save_pretrained(os.path.join(save_dir, name + "_tokenizer"))

    def train(self):
        swanlab.init(
            project="bert-ner1",
            config={
                "learning_rate": self.config["learning_rate"],
                "batch_size": self.config["batch_size"],
                "epochs": self.config["epochs"],
                "max_len": self.config["max_len"],
                "model_name": self.config["model_path"],
                "dropout_rate": self.config.get("dropout_rate", 0.1),
                "align_type": self.config.get("align_type", "ignore"),
                "num_labels": len(self.label2id)
            }
        )

        optimizer = AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            count = 0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = labels.reshape(-1)
                mask_flat = attention_mask.reshape(-1) == 1
                flat_logits = logits_flat[mask_flat]
                flat_labels = labels_flat[mask_flat]
                loss_val = loss_fn(flat_logits, flat_labels)

                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss = total_loss + loss_val.item()
                count = count + 1

            avg_train_loss = total_loss / count

            eval_loss, eval_f1, eval_precision, eval_recall, all_labels, all_preds = self.eval()

            print("第", epoch + 1, "轮")
            print("训练loss:", avg_train_loss)
            print("验证集 Precision:", eval_precision, "Recall:", eval_recall, "F1:", eval_f1)

            swanlab.log({
                "train/loss": avg_train_loss,
                "eval/loss": eval_loss,
                "eval/precision": eval_precision,
                "eval/recall": eval_recall,
                "eval/f1": eval_f1,
                "epoch": epoch + 1
            })

            if eval_f1 > self.best_f1:
                self.best_f1 = eval_f1
                self.save_model()
                print("保存最佳模型，F1:", eval_f1)
                swanlab.log({
                    "best/f1": eval_f1,
                    "best/epoch": epoch + 1
                })

        print("最佳F1:", self.best_f1)
        self.test()
        swanlab.finish()

    def eval(self):
        self.model.eval()
        total_loss = 0
        count = 0
        all_preds = []
        all_labels = []
        loss_fct = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.dev_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = labels.reshape(-1)
                mask_flat = attention_mask.reshape(-1) == 1
                valid_logits = logits_flat[mask_flat]
                valid_labels = labels_flat[mask_flat]
                loss = loss_fct(valid_logits, valid_labels)

                total_loss = total_loss + loss.item()
                count = count + 1
                preds = torch.argmax(logits, dim=-1)

                batch_preds, batch_labels = decode_predict(input_ids, preds, labels, attention_mask, self.id2label)
                all_preds.extend(batch_preds)
                all_labels.extend(batch_labels)

        avg_loss = total_loss / count
        f1, precision, recall = self.evaluator.evaluate(all_labels, all_preds)

        return avg_loss, f1, precision, recall, all_labels, all_preds

    def test(self):
        if not os.path.exists(self.config["save_path"]):
            return

        self.model.load_state_dict(torch.load(self.config["save_path"], map_location=self.device))
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=-1)

                batch_preds, batch_labels = decode_predict(input_ids, preds, labels, attention_mask, self.id2label)
                all_preds.extend(batch_preds)
                all_labels.extend(batch_labels)

        self.evaluator.print_report(all_labels, all_preds)

        f1, precision, recall = self.evaluator.evaluate(all_labels, all_preds)
        print("测试集:", precision, "Recall:", recall, "F1:", f1)

        swanlab.log({
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1
        })

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