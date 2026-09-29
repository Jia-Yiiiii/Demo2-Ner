import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NEREvaluator:

    def extract_entities(self, labels):
        entities = []
        i = 0
        while i < len(labels):
            if labels[i].startswith('B-'):
                entity_type = labels[i][2:]
                start = i
                i += 1
                while i < len(labels) and labels[i].startswith('I-') and labels[i][2:] == entity_type:
                    i += 1
                entities.append((entity_type, start, i - 1))
            else:
                i += 1
        return set(entities)

    def compute_metrics(self, y_true, y_pred):
        total_tp = 0
        total_pred = 0
        total_true = 0
        for true_seq, pred_seq in zip(y_true, y_pred):
            true_entities = self.extract_entities(true_seq)
            pred_entities = self.extract_entities(pred_seq)
            tp = len(true_entities & pred_entities)
            total_tp += tp
            total_pred += len(pred_entities)
            total_true += len(true_entities)
        eps = 1e-8
        precision = total_tp / (total_pred + eps)
        recall = total_tp / (total_true + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return precision, recall, f1, total_tp, total_pred, total_true

    def evaluate(self, y_true, y_pred):
        p, r, f1, _, _, _ = self.compute_metrics(y_true, y_pred)
        return f1, p, r

    def print_report(self, y_true, y_pred):
        type_tp = {}
        type_pred = {}
        type_true = {}
        for i in range(len(y_true)):
            true_entities = self.extract_entities(y_true[i])
            pred_entities = self.extract_entities(y_pred[i])
            for entity in pred_entities:
                label = entity[0]
                type_pred[label] = type_pred.get(label, 0) + 1
                if entity in true_entities:
                    type_tp[label] = type_tp.get(label, 0) + 1
            for entity in true_entities:
                label = entity[0]
                type_true[label] = type_true.get(label, 0) + 1

        p, r, f1, _, _, _ = self.compute_metrics(y_true, y_pred)

        all_labels = []
        for label in type_true.keys():
            if label not in all_labels:
                all_labels.append(label)
        for label in type_pred.keys():
            if label not in all_labels:
                all_labels.append(label)
        all_labels.sort()

        print("类型    精确率  召回率  F1  样本数")
        for label in all_labels:
            tp_c = type_tp.get(label, 0)
            pred_c = type_pred.get(label, 0)
            true_c = type_true.get(label, 0)
            p_c = tp_c / pred_c if pred_c > 0 else 0
            r_c = tp_c / true_c if true_c > 0 else 0
            f_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0
            print(label, round(p_c, 4), round(r_c, 4), round(f_c, 4), true_c)

        print("micro", round(p, 4), round(r, 4), round(f1, 4), sum(type_true.values()))

        macro_p_sum = 0
        macro_r_sum = 0
        macro_f_sum = 0
        count = 0
        for label in all_labels:
            tp_c = type_tp.get(label, 0)
            pred_c = type_pred.get(label, 0)
            true_c = type_true.get(label, 0)
            p_c = tp_c / pred_c if pred_c > 0 else 0
            r_c = tp_c / true_c if true_c > 0 else 0
            f_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0
            macro_p_sum += p_c
            macro_r_sum += r_c
            macro_f_sum += f_c
            count += 1

        macro_p = macro_p_sum / count if count > 0 else 0
        macro_r = macro_r_sum / count if count > 0 else 0
        macro_f = macro_f_sum / count if count > 0 else 0
        print("macro", round(macro_p, 4), round(macro_r, 4), round(macro_f, 4), sum(type_true.values()))


def decode_predict(input_ids, preds, labels, attention_mask, id2label):
    all_pred = []
    all_label = []
    for i in range(len(input_ids)):
        ps = []
        ls = []
        L = attention_mask[i].sum().item()
        for j in range(L):
            if labels[i][j].item() != -100:
                ps.append(id2label.get(preds[i][j].item(), 'O'))
                ls.append(id2label.get(labels[i][j].item(), 'O'))
        all_pred.append(ps)
        all_label.append(ls)
    return all_pred, all_label