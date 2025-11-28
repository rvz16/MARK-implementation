import glob
import os

import torch

from mark.backbones import MAGIEncoder, DMoNEncoder
from mark.datasets import load_tag
from mark.metrics import compute_all, save_metrics
from mark.utils import load_config, setup_logger, get_device


def latest_mark_run(log_dir: str, dataset: str):
    """Find the latest MARK run directory."""
    base = os.path.join(log_dir, dataset)
    if not os.path.exists(base):
        return None
    candidates = [d for d in glob.glob(os.path.join(base, "mark-*")) if os.path.isdir(d)]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def main():
    data_cfg = load_config("configs/data.yaml")
    engine_cfg = load_config("configs/engine.yaml")
    train_cfg = load_config("configs/train.yaml")
    logger = setup_logger()
    device = get_device(train_cfg.get("device", "cpu"))

    dataset = data_cfg.get("dataset", "cora")
    data_dir = data_cfg.get("data_dir", "./data")
    log_dir = train_cfg.get("log_dir", "./experiments")

    run_dir = latest_mark_run(log_dir, dataset)
    if not run_dir:
        raise RuntimeError("No mark run found. Run scripts/run_mark.py first.")
    
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint_step*.pt")))
    ckpt_file = ckpts[-1] if ckpts else os.path.join(run_dir, "checkpoint.pt")
    logger.info(f"Evaluating checkpoint {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu")

    x, edge_index, texts, y, K_true = load_tag(dataset, data_dir)
    K = engine_cfg.get("num_clusters", 0) or K_true
    in_dim = x.size(1)
    hidden_dim = engine_cfg.get("hidden_dim", 256)
    proj_dim = engine_cfg.get("proj_dim", 128)
    backbone_name = engine_cfg.get("backbone", "magi").lower()
    
    if backbone_name == "magi":
        model = MAGIEncoder(
            in_dim, hidden_dim, proj_dim, K, 
            engine_cfg.get("tau_align", 0.5), 
            engine_cfg.get("lambda_clu", 1.0),
            num_layers=engine_cfg.get("num_layers", 2),
            dropout=engine_cfg.get("dropout", 0.0),
        )
    else:
        model = DMoNEncoder(
            in_dim, hidden_dim, K,
            num_layers=engine_cfg.get("num_layers", 2),
            collapse_regularization=engine_cfg.get("collapse_reg", 0.1),
            dropout=engine_cfg.get("dropout", 0.0),
        )
    
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    x = x.to(device)
    edge_index = edge_index.to(device)

    with torch.no_grad():
        if backbone_name == "magi":
            z = model(x, edge_index)
            assign = model.predict_assign(z).cpu()
        else:
            h, S = model(x, edge_index)
            assign = model.predict_assign((h, S)).cpu()
    
    metrics = compute_all(y.numpy(), assign.numpy())
    metrics_path_json = os.path.join(run_dir, "metrics.json")
    metrics_path_csv = os.path.join(run_dir, "metrics.csv")
    save_metrics(metrics, metrics_path_json, metrics_path_csv)
    
    logger.info(f"Results for {dataset}:")
    logger.info(f"  ACC={metrics['ACC']:.4f}")
    logger.info(f"  NMI={metrics['NMI']:.4f}")
    logger.info(f"  ARI={metrics['ARI']:.4f}")
    logger.info(f"  F1={metrics['F1']:.4f}")
    logger.info(f"Saved metrics to {metrics_path_json}")


if __name__ == "__main__":
    main()
