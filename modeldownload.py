import os
from modelscope import snapshot_download


def download_model(model_id: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        model_id=model_id,
        cache_dir=local_dir,
        revision="master"
    )
    print(f"下载完成: {local_dir}\n")


if __name__ == "__main__":
    models = [
        ("bert-base-chinese", "./bert-base-chinese"),
        ("hfl/chinese-bert-wwm", "./hfl-chinese-bert-wwm"),
    ]

    for model_id, local_dir in models:
        download_model(model_id, local_dir)

    print("全部模型下载完成！")
