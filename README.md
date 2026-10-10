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
    print(f"正在下载: {filename}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_folder,
        # force_download=True, # 初次下载注释掉，避免重复下载
        force_download=False,
        local_dir_use_symlinks=False
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

| 类型 | 精确率 | 召回率 | F1 | 样本数 |
| :--- | :--- | :--- | :--- | :--- |
| LOC | 0.9377 | 0.9051 | 0.9211 | 632 |
| ORG | 0.8264 | 0.8881 | 0.8561 | 268 |
| PER | 0.9449 | 0.9501 | 0.9475 | 361 |
| micro | 0.9144 | 0.9144 | 0.9144 | 1261 |
| macro | 0.9030 | 0.9144 | 0.9082 | 1261 |



#### 训练曲线

<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/651aceaa-8724-463c-981b-8d2130600634" />
  <p style="font-size: 13px; color:#666;">图 4-1 训练曲线 1</p>
</div>

<div align="center">
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/48fa3098-98af-4306-8582-0c33253431b8" />
  <p style="font-size: 13px; color:#666;">图 4-2 训练曲线 2</p>
</div>

<div align="center">
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/47f73a7a-2254-4366-85db-e860f8ca5484" />
  <p style="font-size: 13px; color:#666;">图 4-3 训练曲线 3</p>
</div>

<div align="center">
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/a880cc2d-ca49-49b2-842b-53d9a1fda4d1" />
  <p style="font-size: 13px; color:#666;">图 4-4 训练曲线 4</p>
</div>



---

#### (2) chinese-bert-wwm

运行命令：
```bash
python trainer.py configs/Bert_Config_exp5.json
```

测试集结果：
| 类型 | 精确率 | 召回率 | F1 | 样本数 |
| :--- | :--- | :--- | :--- | :--- |
| LOC | 0.9413 | 0.913 | 0.9269 | 632 |
| ORG | 0.8561 | 0.8881 | 0.8718 | 268 |
| PER | 0.9766 | 0.9252 | 0.9502 | 361 |
| micro | 0.9319 | 0.9112 | 0.9214 | 1261 |
| macro | 0.9247 | 0.9087 | 0.9163 | 1261 |



训练曲线：
<img width="532" height="297" alt="image" src="https://github.com/user-attachments/assets/5d59c1c9-f304-4eab-acba-d6b2f82ab39e" />
<img width="1571" height="591" alt="image" src="https://github.com/user-attachments/assets/16ad5b5e-7f41-4c64-b439-c8777941a56a" />
<img width="1060" height="641" alt="image" src="https://github.com/user-attachments/assets/4c92fce3-0e8d-496e-b417-499f3d8844e7" />
<img width="1576" height="302" alt="image" src="https://github.com/user-attachments/assets/8de7bd09-ffce-401a-94fe-c8ca912e5ebb" />




---

### 2.2 Weibo 数据集

#### (1) bert-base-chinese

运行命令：
```bash
python trainer.py configs/Bert_Config_exp1.json
```

测试集结果：

| 类型 | 精确率 | 召回率 | F1 | 样本数 |
| :--- | :--- | :--- | :--- | :--- |
| GPE. NAM | 0.7843 | 0.8696 | 0.8247 | 46 |
| GPE. NOM | 0.0 | 0.0 | 0.0 | 2 |
| LOC. NAM | 0.2778 | 0.2632 | 0.2703 | 19 |
| LOC. NOM | 0.4167 | 0.5556 | 0.4762 | 9 |
| ORG. NAM | 0.5128 | 0.5128 | 0.5128 | 39 |
| ORG. NOM | 0.5 | 0.5 | 0.5 | 16 |
| PER. NAM | 0.7545 | 0.7545 | 0.7545 | 110 |
| PER. NOM | 0.6811 | 0.7545 | 0.7159 | 167 |
| micro | 0.6659 | 0.7034 | 0.6841 | 408 |
| macro | 0.4909 | 0.5263 | 0.5068 | 408 |



训练曲线：

<img width="520" height="296" alt="image" src="https://github.com/user-attachments/assets/8d0735be-21d6-4f48-a3f8-658dba8dec25" />
<img width="1610" height="597" alt="image" src="https://github.com/user-attachments/assets/2f0d1b38-adba-4f07-b492-139ca6605f83" />
<img width="1065" height="635" alt="image" src="https://github.com/user-attachments/assets/fdeb4a17-7dbc-4290-a3c1-36cee6aa44fc" />
<img width="1582" height="307" alt="image" src="https://github.com/user-attachments/assets/44cebc81-0f44-4064-b62d-50ba36560d4b" />




---

#### (2) chinese-bert-wwm (ignore)

运行命令：
```bash
python trainer.py configs/Bert_Config_exp2.json
```

测试集结果：

| 类型 | 精确率 | 召回率 | F1 | 样本数 |
| :--- | :--- | :--- | :--- | :--- |
| GPE. NAM | 0.7091 | 0.8478 | 0.7723 | 46 |
| GPE. NOM | 0.0 | 0.0 | 0.0 | 2 |
| LOC. NAM | 0.45 | 0.4737 | 0.4615 | 19 |
| LOC. NOM | 0.2727 | 0.3333 | 0.3 | 9 |
| ORG. NAM | 0.5588 | 0.4872 | 0.5205 | 39 |
| ORG. NOM | 0.5 | 0.4375 | 0.4667 | 16 |
| PER. NAM | 0.7652 | 0.8 | 0.7822 | 110 |
| PER. NOM | 0.7056 | 0.7605 | 0.732 | 167 |
| micro | 0.6807 | 0.7157 | 0.6977 | 408 |
| macro | 0.4952 | 0.5175 | 0.5044 | 408 |



训练曲线：

<img width="536" height="292" alt="image" src="https://github.com/user-attachments/assets/0ff7803c-a487-4707-a3a4-820c2dd96ce8" />
<img width="1566" height="597" alt="image" src="https://github.com/user-attachments/assets/aa5dcc17-6dba-4093-8f7a-c053385fc488" />
<img width="1055" height="638" alt="image" src="https://github.com/user-attachments/assets/0ffcf4cb-3add-4c07-80b3-6f344b78aecf" />
<img width="1576" height="311" alt="image" src="https://github.com/user-attachments/assets/8eeb9b01-f994-48dd-9813-3bffd32b1c4b" />

---

#### (3) chinese-bert-wwm (same)

运行命令：
```bash
python trainer.py configs/Bert_Config_exp3.json
```

测试集结果：
| 类型 | 精确率 | 召回率 | F1 | 样本数 |
| :--- | :--- | :--- | :--- | :--- |
| GPE. NAM | 0.7455 | 0.8913 | 0.8119 | 46 |
| GPE. NOM | 0.0 | 0.0 | 0.0 | 2 |
| LOC. NAM | 0.5 | 0.3684 | 0.4242 | 19 |
| LOC. NOM | 0.4286 | 0.3333 | 0.375 | 9 |
| ORG. NAM | 0.5625 | 0.4615 | 0.507 | 39 |
| ORG. NOM | 0.5625 | 0.5625 | 0.5625 | 16 |
| PER. NAM | 0.72 | 0.6545 | 0.6857 | 110 |
| PER. NOM | 0.6538 | 0.7126 | 0.6819 | 167 |
| micro | 0.6626 | 0.6593 | 0.6609 | 408 |
| macro | 0.5216 | 0.4980 | 0.5060 | 408 |


训练曲线：
<img width="540" height="295" alt="image" src="https://github.com/user-attachments/assets/d63ab067-919c-4ed2-a5dd-b0ddaaab340f" />
<img width="1573" height="597" alt="image" src="https://github.com/user-attachments/assets/98432cc0-d966-44e7-9e0b-4ce2fa38af2a" />
<img width="1052" height="623" alt="image" src="https://github.com/user-attachments/assets/902c2d4f-13c5-467b-97be-6d45b47cd448" />
<img width="1576" height="307" alt="image" src="https://github.com/user-attachments/assets/62444637-8644-466b-b473-6a89964a8ac4" />



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
├── data_process.py
├── model.py
├── trainer.py
├── utils.py
├── requirements.txt
└── README.md
```
