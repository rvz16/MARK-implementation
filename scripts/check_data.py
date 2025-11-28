import os
import time
import urllib.request

import torch

from mark.datasets import load_tag
from mark.utils import load_config, setup_logger


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)
    return total


def check_internet():
    try:
        urllib.request.urlopen("https://pytorch.org", timeout=3)
        return True
    except Exception:
        return False


def main():
    logger = setup_logger()
    data_cfg = load_config("configs/data.yaml")
    dataset = data_cfg.get("dataset", "cora")
    data_dir = data_cfg.get("data_dir", "./data")
    logger.info(f"Checking dataset {dataset} in {data_dir}")

    start = time.perf_counter()
    x, edge_index, texts, y, K = load_tag(dataset, data_dir)
    elapsed = time.perf_counter() - start

    logger.info(f"Feature shape: {x.shape}, edge_index: {edge_index.shape}, labels: {y.shape}, K={K}")
    logger.info(f"First texts: {texts[:3]}")
    logger.info(f"Download/load time: {elapsed:.2f}s")

    size = dir_size_bytes(os.path.abspath(data_dir))
    logger.info(f"Approx dataset directory bytes: {size}")
    logger.info(f"Internet access: {check_internet()}")
    logger.info(f"TORCH_GEOMETRIC_HOME={os.environ.get('TORCH_GEOMETRIC_HOME')}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
