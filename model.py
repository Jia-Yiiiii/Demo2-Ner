from transformers import BertModel
import torch.nn as nn


class BertForNER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config["model_path"])
        self.dropout = nn.Dropout(config.get("dropout_rate", 0.1))
        hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(hidden_size, config["num_labels"])

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out = self.dropout(out.last_hidden_state)
        logits = self.fc(out)
        return logits