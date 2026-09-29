import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class NERDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = config['max_len']
        self.align_type = config.get('align_type', 'ignore')
        self.data = self.read_data(config['data_path'])
        self.label2id, self.id2label = self.get_label_map(config)
        self.raw_data = self.data

    def read_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        for sentence in content.strip().split('\n\n'):
            if not sentence:
                continue
            tokens, labels = [], []
            for line in sentence.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        tokens.append(parts[0])
                        labels.append(parts[1] if parts[1] != '0' else 'O')
            if tokens:
                data.append((tokens, labels))
        return data

    def get_label_map(self, config):

        if 'label2id' in config and config['label2id'] is not None:
            label2id = config['label2id']
            id2label = {}
            for k, v in label2id.items():
                id2label[v] = k
            return label2id, id2label

        labels = {'O'}
        for _, tag_list in self.data:
            for tag in tag_list:
                labels.add(tag)

        labels = sorted(labels)
        label2id = {}
        id2label = {}
        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label

        return label2id, id2label

    def align_labels(self, word_ids, tags):

        label_ids = []
        prev_wid = -1
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
                continue
            if wid != prev_wid:
                prev_wid = wid
                label = tags[wid] if wid < len(tags) else 'O'
                label_ids.append(self.label2id.get(label, -100))
            else:
                if self.align_type == 'same':
                    label = tags[wid] if wid < len(tags) else 'O'
                    if label.startswith('B-'):
                        label = 'I-' + label[2:]
                    label_ids.append(self.label2id.get(label, -100))
                else:
                    label_ids.append(-100)

        return label_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        words, tags = self.data[idx]
        return words, tags

    def collate_fn(self, batch):
        words_list = []
        tags_list = []
        for words, tags in batch:
            words_list.append(words)
            tags_list.append(tags)

        tokenized = self.tokenizer(
            words_list,
            is_split_into_words=True,
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        labels = []
        for idx in range(len(batch)):
            word_ids = tokenized.word_ids(idx)
            label_ids = self.align_labels(word_ids, tags_list[idx])
            labels.append(torch.tensor(label_ids, dtype=torch.long))

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )

        return tokenized['input_ids'], tokenized['attention_mask'], labels

    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )