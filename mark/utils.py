import json
import logging
import os
import random
import time
from contextlib import contextmanager, nullcontext
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cpu") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logger(name: str = "MARK", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


@contextmanager
def timed(logger: logging.Logger, name: str):
    start = time.perf_counter()
    logger.info(f"[TIME] {name} start")
    yield
    end = time.perf_counter()
    logger.info(f"[TIME] {name} done in {end - start:.3f}s")


def maybe_autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.amp.autocast('cuda')
    return nullcontext()


def save_config_snapshot(configs: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(configs, f)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_plm_cache(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        return torch.load(path)
    return {}


def save_plm_cache(cache: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(cache, path)


class TokenCostTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_requests = 0

    def add_usage(self, usage: Optional[Dict[str, Any]]):
        if not usage:
            return
        tokens = usage.get("total_tokens") or usage.get("completion_tokens") or 0
        self.total_tokens += int(tokens)
        self.total_requests += 1

    def to_dict(self) -> Dict[str, Any]:
        return {"total_tokens": self.total_tokens, "total_requests": self.total_requests}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
