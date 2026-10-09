# 基于 BERT 的中文命名实体识别

本项目使用 BERT 模型在 MSRA 和 Weibo 两个中文数据集上进行命名实体识别（NER）实验，并对比了不同模型和标签对齐策略的效果。

---

## 预训练模型本地下载与加载

本项目采用**本地路径加载预训练权重**。若直接调用 `from_pretrained("模型名称")`，`transformers` 库默认自动下载缓存文件，会同时拉取 PyTorch、TensorFlow 等多框架权重，存在大量冗余文件。手动下载仅保留 PyTorch 运行必需文件，磁盘占用更小，保证运行环境统一。

项目使用两组预训练模型：
- `bert-base-chinese`
- `hfl/chinese-bert-wwm`


```bash


# 下载 bert-base-chinese
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

repo_id = "bert-base-chinese"
local_folder = "./bert-base-chinese"
os.makedirs(local_folder, exist_ok=True)

file_list = [
    "config.json",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "pytorch_model.bin"
]

for filename in file_list:

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_folder,
        force_download=True
    )
print("全部文件下载完成！")

# 下载 hfl/chinese-bert-wwm
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

repo_id = "hfl/chinese-bert-wwm"
local_folder = "./hfl-chinese-bert-wwm"
os.makedirs(local_folder, exist_ok=True)

file_list = [
    "config.json",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "pytorch_model.bin"
]

for filename in file_list:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_folder
    )
print("全部文件下载完成！")
```

---

## 一、数据分析

### 1.1 数据格式

数据加载通过 `split('\n\n')` 按空行分隔句子；标签中的 `'0'` 在读取时自动转换为 `'O'`。标签映射优先从配置读取预定义的 `label2id`，若不存在则自动从数据中提取所有标签建立映射。子词对齐通过 `align_type` 参数控制，默认为 `'ignore'` 将非首子词标签设为 `-100`，若设为 `'same'` 则对后续子词使用 `I-` 标签。

两个数据集的标签体系不同：

| 数据集 | 标签格式 | 标签数量 |
|--------|---------|---------|
| MSRA | `B-LOC` 形式 | 7 种 |
| Weibo | `B-LOC.NAM` / `B-LOC.NOM` | 17 种 |

代码为两个数据集分别建立独立的 `label2id` 映射，不共用标签体系。

---

### 1.2 标签分布

**MSRA 训练集标签分布：**

| 标签 | 数量 |
|------|------|
| O | 206,412 |
| I-ORG | 9,141 |
| I-LOC | 5,313 |
| B-LOC | 3,952 |
| I-PER | 3,612 |
| B-ORG | 2,158 |
| B-PER | 1,850 |

**Weibo 训练集标签分布：**

| 标签 | 数量 |
|------|------|
| O | 68,777 |
| I-PER.NOM | 1,043 |
| I-PER.NAM | 1,041 |
| B-PER.NOM | 766 |
| B-PER.NAM | 574 |
| I-ORG.NAM | 477 |
| I-GPE.NAM | 241 |
| B-GPE.NAM | 205 |
| B-ORG.NAM | 183 |
| I-LOC.NAM | 129 |
| I-LOC.NOM | 66 |
| I-ORG.NOM | 61 |
| B-LOC.NAM | 56 |
| B-LOC.NOM | 51 |
| B-ORG.NOM | 42 |
| B-GPE.NOM | 8 |
| I-GPE.NOM | 8 |

---

### 1.3 数据处理流程

| 步骤 | 操作 | 说明 |
|:---:|:---:|---|
| 1 | 数据读取 | 通过 `split('\n\n')` 按空行分隔句子，`'0'` 自动转为 `'O'` |
| 2 | 标签映射 | 优先从配置读取预定义 `label2id`，否则自动从数据中提取 |
| 3 | 子词对齐 | 使用 `word_ids` 对齐，由 `align_type` 参数控制策略 |

**对齐策略说明：**

| 策略 | 说明 |
|:---:|---|
| `ignore` | 只保留词首子词的标签，其余子词忽略（设为 `-100`） |
| `same` | 词首子词用原始标签，后续子词复制标签（`B-` 转为 `I-`） |

---

## 二、实验结果

### 2.1 MSRA 数据集

#### (1) bert-base-chinese

运行命令：
```bash
python trainer.py configs/Bert_Config_exp4.json
```

测试集结果：

| 类型    | 精确率 | 召回率 | F1     | 样本数 |
|---------|--------|--------|--------|--------|
| LOC     | 0.9405 | 0.9003 | 0.9200 | 632    |
| ORG     | 0.8478 | 0.9142 | 0.8797 | 268    |
| PER     | 0.9586 | 0.9612 | 0.9599 | 361    |
| micro   | 0.9244 | 0.9207 | 0.9225 | 1261   |
| macro   | 0.9156 | 0.9252 | 0.9199 | 1261   |

训练曲线：

<img width="580" height="301" alt="image" src="https://github.com/user-attachments/assets/48d00b33-aac0-476e-a15a-4a788fa61e1d" />

<img width="1582" height="602" alt="image" src="https://github.com/user-attachments/assets/3ca5b1ab-4afa-4e58-b0c8-c7059944af39" />

<img width="1582" height="640" alt="image" src="https://github.com/user-attachments/assets/764598f0-f7af-4970-8447-55aaffb32d5a" />



实验日志：https://swanlab.cn/@2225/bert-ner1/runs/x9w7la1y/overview

---

#### (2) chinese-bert-wwm

运行命令：
```bash
python trainer.py configs/Bert_Config_exp5.json
```

测试集结果：

| 类型    | 精确率 | 召回率 | F1     | 样本数 |
|---------|--------|--------|--------|--------|
| LOC     | 0.9536 | 0.9098 | 0.9312 | 632    |
| ORG     | 0.8445 | 0.8918 | 0.8675 | 268    |
| PER     | 0.9474 | 0.9474 | 0.9474 | 361    |
| micro   | 0.9270 | 0.9167 | 0.9219 | 1261   |
| macro   | 0.9152 | 0.9163 | 0.9154 | 1261   |

训练曲线：

<img width="535" height="295" alt="image" src="https://github.com/user-attachments/assets/3ef577ad-a4b7-473a-996d-669a468fa1a9" />
<img width="1562" height="597" alt="image" src="https://github.com/user-attachments/assets/5148458b-7a27-4bc7-8075-17ac9a8817c2" />
<img width="1572" height="635" alt="image" src="https://github.com/user-attachments/assets/bdd76291-d203-445c-8395-feec915a1a81" />


实验日志：https://swanlab.cn/@2225/bert-ner1/runs/ov43sjgr/overview

---

### 2.2 Weibo 数据集

#### (1) bert-base-chinese

运行命令：
```bash
python trainer.py configs/Bert_Config_exp1.json
```

测试集结果：

| 类型       | 精确率   | 召回率   | F1       | 样本数 |
|------------|----------|----------|----------|--------|
| GPE.NAM    | 0.7358   | 0.8478   | 0.7879   | 46     |
| GPE.NOM    | 0.0000   | 0.0000   | 0.0000   | 2      |
| LOC.NAM    | 0.3103   | 0.4737   | 0.3750   | 19     |
| LOC.NOM    | 0.3636   | 0.4444   | 0.4000   | 9      |
| ORG.NAM    | 0.7500   | 0.3077   | 0.4364   | 39     |
| ORG.NOM    | 0.5714   | 0.5000   | 0.5333   | 16     |
| PER.NAM    | 0.7080   | 0.7273   | 0.7175   | 110    |
| PER.NOM    | 0.7126   | 0.7126   | 0.7126   | 167    |
| micro      | 0.6725   | 0.6642   | 0.6683   | 408    |
| macro      | 0.5190   | 0.5017   | 0.4953   | 408    |

训练曲线：

<img width="523" height="301" alt="image" src="https://github.com/user-attachments/assets/fc026ddd-3821-4ca7-b38c-c5987fccbd1b" />

<img width="1571" height="588" alt="image" src="https://github.com/user-attachments/assets/1b7d0480-8109-464f-97aa-edc8f414318e" />

<img width="1575" height="636" alt="image" src="https://github.com/user-attachments/assets/b08c89b6-e996-4836-bd9b-0aaa7c00664a" />


实验日志：https://swanlab.cn/@2225/bert-ner1/runs/wh0taazt/overview

---

#### (2) chinese-bert-wwm (ignore)

运行命令：
```bash
python trainer.py configs/Bert_Config_exp2.json
```

测试集结果：

| 类型       | 精确率   | 召回率   | F1       | 样本数 |
|------------|----------|----------|----------|--------|
| GPE.NAM    | 0.7455   | 0.8913   | 0.8119   | 46     |
| GPE.NOM    | 0.0000   | 0.0000   | 0.0000   | 2      |
| LOC.NAM    | 0.3684   | 0.3684   | 0.3684   | 19     |
| LOC.NOM    | 0.4286   | 0.3333   | 0.3750   | 9      |
| ORG.NAM    | 0.4211   | 0.4103   | 0.4156   | 39     |
| ORG.NOM    | 0.7143   | 0.6250   | 0.6667   | 16     |
| PER.NAM    | 0.7241   | 0.7636   | 0.7434   | 110    |
| PER.NOM    | 0.7024   | 0.7066   | 0.7045   | 167    |
| micro      | 0.6691   | 0.6838   | 0.6764   | 408    |
| macro      | 0.5130   | 0.5123   | 0.5107   | 408    |

训练曲线：

<img width="523" height="290" alt="image" src="https://github.com/user-attachments/assets/ce1fbcc0-171a-4777-9cc1-c93be3dc8ad6" />
<img width="1565" height="582" alt="image" src="https://github.com/user-attachments/assets/e03ecbc4-6b6f-49cf-ae65-357bab117991" />
<img width="1577" height="637" alt="image" src="https://github.com/user-attachments/assets/12f2f900-7f94-47cd-9b8a-fe95e0f0a7c1" />



实验日志：https://swanlab.cn/@2225/bert-ner1/runs/zdet0alr/overview
---

#### (3) chinese-bert-wwm (same)

运行命令：
```bash
python trainer.py configs/Bert_Config_exp3.json
```

测试集结果：
| 类型       | 精确率   | 召回率   | F1       | 样本数 |
|------------|----------|----------|----------|--------|
| GPE.NAM    | 0.7736   | 0.8913   | 0.8283   | 46     |
| GPE.NOM    | 0.0000   | 0.0000   | 0.0000   | 2      |
| LOC.NAM    | 0.3750   | 0.3158   | 0.3429   | 19     |
| LOC.NOM    | 0.2727   | 0.3333   | 0.3000   | 9      |
| ORG.NAM    | 0.5000   | 0.4615   | 0.4800   | 39     |
| ORG.NOM    | 0.5294   | 0.5625   | 0.5455   | 16     |
| PER.NAM    | 0.7358   | 0.7091   | 0.7222   | 110    |
| PER.NOM    | 0.6919   | 0.7126   | 0.7021   | 167    |
| micro      | 0.6650   | 0.6716   | 0.6683   | 408    |
| macro      | 0.4848   | 0.4983   | 0.4901   | 408    |

训练曲线：

<img width="567" height="297" alt="image" src="https://github.com/user-attachments/assets/437fc200-5bec-4ba8-83c4-6567be2f1765" />

<img width="1560" height="597" alt="image" src="https://github.com/user-attachments/assets/82a50160-cc52-4bec-b110-90990d127adf" />

<img width="1580" height="637" alt="image" src="https://github.com/user-attachments/assets/6a399fa9-3ccf-4260-9b92-f9dd8f21ca4f" />


实验日志：https://swanlab.cn/@2225/bert-ner1/runs/uowccrdd/overview

---

## 三、项目结构

```text
BERT-NER-DEMO2/
├── data/
│   ├── MSRA/
│   │   ├── train.txt
│   │   ├── dev.txt
│   │   └── test.txt
│   └── weibo/
│       ├── train.txt
│       ├── dev.txt
│       └── test.txt
├── configs/
│   ├── Bert_Config_exp1.json
│   ├── Bert_Config_exp2.json
│   ├── Bert_Config_exp3.json
│   ├── Bert_Config_exp4.json
│   ├── Bert_Config_exp5.json
│   └── label2id.json
├── data.py
├── model.py
├── trainer.py
├── utils.py
├── requirements.txt
└── README.md
```
