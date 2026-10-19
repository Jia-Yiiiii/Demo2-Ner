import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

models = [
    ("bert-base-chinese", "./bert-base-chinese"),
    ("hfl/chinese-bert-wwm", "./hfl-chinese-bert-wwm"),
]

# bert-base-chinese 有 tokenizer.json，hfl/chinese-bert-wwm 没有
file_lists = {
    "bert-base-chinese": [
        "config.json", "vocab.txt", "tokenizer.json",
        "tokenizer_config.json", "pytorch_model.bin"
    ],
    "hfl/chinese-bert-wwm": [
        "config.json", "vocab.txt",
        "tokenizer_config.json", "pytorch_model.bin"
    ],
}

for repo_id, local_folder in models:
    os.makedirs(local_folder, exist_ok=True)
    for filename in file_lists[repo_id]:
        print(f"正在下载: {repo_id}/{filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_folder,
            force_download=False,
        )
    print(f"{repo_id} 下载完成\n")

print("全部模型下载完成！")
