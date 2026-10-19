# 基于 BERT 的中文命名实体识别

本项目使用 BERT 模型在 MSRA 和 Weibo 两个中文数据集上进行命名实体识别（NER）实验，并对比了不同模型和标签对齐策略的效果。

---

## 预训练模型本地下载与加载
```bash
python modeldownload.py
```
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
<img width="640" height="251" alt="7c4088645c68611e7155c7a54ea8da8d" src="https://github.com/user-attachments/assets/2611606a-873a-4d98-9d5a-b3ec85f76eb2" />


#### 训练曲线


<img width="516" height="352" alt="image" src="https://github.com/user-attachments/assets/0880058d-ae21-470f-85f6-df6f06b68be1" />





<img width="1520" height="717" alt="image" src="https://github.com/user-attachments/assets/49c064db-a510-4a45-a5fc-e47e2c1415d7" />






<img width="1533" height="732" alt="image" src="https://github.com/user-attachments/assets/c65528a5-bf4e-48a3-95eb-8def1f518010" />






---

#### (2) chinese-bert-wwm

运行命令：
```bash
python trainer.py configs/Bert_Config_exp5.json
```

测试集结果：
<img width="625" height="247" alt="0ec2d9050992c0bf475f36bf92bc783c" src="https://github.com/user-attachments/assets/91d6efb3-b8e5-4227-b421-af8d787009a9" />


#### 训练曲线

<img width="517" height="340" alt="image" src="https://github.com/user-attachments/assets/73d304b2-0b68-4717-b23b-ed188f5242e3" />





<img width="1541" height="647" alt="image" src="https://github.com/user-attachments/assets/6c80fdff-5baf-416f-a975-e52da8c2a804" />





<img width="1530" height="716" alt="image" src="https://github.com/user-attachments/assets/8c80b0d1-fbc4-49b7-a079-eb97fa9c0e9a" />







---

### 2.2 Weibo 数据集

#### (1) bert-base-chinese

运行命令：
```bash
python trainer.py configs/Bert_Config_exp1.json
```

测试集结果：
<img width="662" height="367" alt="85d3292410e7e820e7009a86bbc462bc" src="https://github.com/user-attachments/assets/282501f9-ba5c-4392-9c8a-71f063f90707" />


#### 训练曲线

<img width="510" height="340" alt="image" src="https://github.com/user-attachments/assets/a7ba94a8-63f5-4c39-b4c6-165b9a80f351" />




<img width="1535" height="652" alt="image" src="https://github.com/user-attachments/assets/90f9d2ce-7d01-4f30-a880-287a14c67f5f" />




<img width="1540" height="710" alt="image" src="https://github.com/user-attachments/assets/6050eaa5-25c2-4127-97d1-0882a6148551" />








---

#### (2) chinese-bert-wwm (ignore)

运行命令：
```bash
python trainer.py configs/Bert_Config_exp2.json
```

测试集结果：
<img width="632" height="368" alt="720c8a62888f264a46ef6cb2cb933664" src="https://github.com/user-attachments/assets/8977db0f-0f8b-4e66-a8fc-a15479d276a5" />



#### 训练曲线

<img width="508" height="338" alt="image" src="https://github.com/user-attachments/assets/dd8a76a0-1396-4e4a-bc8c-1a5edeeef03f" />




<img width="1540" height="637" alt="image" src="https://github.com/user-attachments/assets/fdac8f0d-a603-488c-b5c5-c2331927bb38" />




<img width="1537" height="716" alt="image" src="https://github.com/user-attachments/assets/4da33e2e-c003-4ea1-a1b5-61f0b2a69973" />






---

#### (3) chinese-bert-wwm (same)

运行命令：
```bash
python trainer.py configs/Bert_Config_exp3.json
```

测试集结果：
<img width="610" height="361" alt="06e94e1d0eb70b1f09d5f10fcfab8248" src="https://github.com/user-attachments/assets/2e7f66d1-a964-4931-a256-fcc258895dd0" />

#### 训练曲线
<img width="522" height="335" alt="image" src="https://github.com/user-attachments/assets/e8a8e31b-1b55-4fe6-bcc3-0ba16aa1100c" />




<img width="1527" height="631" alt="image" src="https://github.com/user-attachments/assets/1dfeec43-85fb-49d7-9227-29476ea50af4" />




<img width="1532" height="721" alt="image" src="https://github.com/user-attachments/assets/9a5bfd51-e0d5-44f2-8d40-2cf7fe8b20a6" />








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
├── modeldownload.py
├── requirements.txt
└── README.md
```
