import os
import time
import json
from datetime import datetime

import torch
from torch import optim
import torch.nn.functional as F

from mark.augment import feature_dropout, edge_dropout
from mark.backbones.magi import MAGIEncoder
from mark.backbones.dmon import DMoNEncoder
from mark.datasets import load_tag
from mark.metrics import compute_all
from mark.utils import (
    set_seed,
    load_config,
    setup_logger,
    get_device,
    maybe_autocast,
    save_config_snapshot,
)


def make_run_dir(base: str, dataset: str, prefix: str) -> str:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(base, dataset, f"{prefix}-{run_id}")
    os.makedirs(path, exist_ok=True)
    return path


def main():
    data_cfg = load_config("configs/data.yaml")
    engine_cfg = load_config("configs/engine.yaml")
    train_cfg = load_config("configs/train.yaml")
    set_seed(train_cfg.get("seed", 42))
    device = get_device(train_cfg.get("device", "cpu"))
    logger = setup_logger()

    dataset = data_cfg.get("dataset", "cora")
    data_dir = data_cfg.get("data_dir", "./data")
    log_dir = train_cfg.get("log_dir", "./experiments")
    run_dir = make_run_dir(log_dir, dataset, "pretrain")
    save_config_snapshot({"data": data_cfg, "engine": engine_cfg, "train": train_cfg}, os.path.join(run_dir, "config.yaml"))

    x, edge_index, texts, y, K = load_tag(dataset, data_dir)
    K_model = engine_cfg.get("num_clusters", 0) or K
    in_dim = x.size(1)
    hidden_dim = engine_cfg.get("hidden_dim", 256)  # Increased default
    proj_dim = engine_cfg.get("proj_dim", 128)      # Increased default

    backbone_name = engine_cfg.get("backbone", "magi").lower()
    if backbone_name == "magi":
        model = MAGIEncoder(
            in_dim, hidden_dim, proj_dim, K_model, 
            engine_cfg.get("tau_align", 0.5), 
            engine_cfg.get("lambda_clu", 1.0),
            num_layers=engine_cfg.get("num_layers", 2),
            dropout=engine_cfg.get("dropout", 0.0),
        )
    elif backbone_name == "dmon":
        model = DMoNEncoder(
            in_dim, hidden_dim, K_model,
            num_layers=engine_cfg.get("num_layers", 2),
            collapse_regularization=engine_cfg.get("collapse_reg", 0.1),
            dropout=engine_cfg.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown backbone {backbone_name}")
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(), 
        lr=train_cfg.get("lr", 1e-3), 
        weight_decay=train_cfg.get("weight_decay", 5e-4)
    )

    x = x.to(device)
    edge_index = edge_index.to(device)
    y_np = y.numpy()

    pretrain_epochs = train_cfg.get("pretrain_epochs", 100)  # Increased default
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    
    # Augmentation parameters
    feat_drop = engine_cfg.get("feat_drop", 0.2)
    edge_drop = engine_cfg.get("edge_drop", 0.2)
    
    timing = []
    best_acc = 0.0
    best_state = None
    
    logger.info(f"Starting pretraining for {pretrain_epochs} epochs on {dataset}")
    logger.info(f"Model: {backbone_name}, hidden={hidden_dim}, proj={proj_dim}, K={K_model}")
    
    for epoch in range(1, pretrain_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()

        # Create augmented views
        x1 = feature_dropout(x, feat_drop)
        x2 = feature_dropout(x, feat_drop)
        e1 = edge_dropout(edge_index, edge_drop)
        e2 = edge_dropout(edge_index, edge_drop)

        with maybe_autocast(device, train_cfg.get("amp", True)):
            if backbone_name == "magi":
                z1 = model(x1, e1)
                z2 = model(x2, e2)
                loss, comps = model.loss_pretrain(z1, z2)
            else:
                h1, S1 = model(x1, e1)
                h2, S2 = model(x2, e2)
                loss1, _ = model.loss_pretrain(S1, e1)
                loss2, _ = model.loss_pretrain(S2, e2)
                loss = 0.5 * (loss1 + loss2)
                comps = {"loss1": float(loss1.item()), "loss2": float(loss2.item())}

        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        dur = time.perf_counter() - epoch_start
        
        # Evaluate metrics periodically
        metrics = {}
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                if backbone_name == "magi":
                    z_eval = model(x, edge_index)
                    pred = model.predict_assign(z_eval).cpu().numpy()
                else:
                    h_eval, S_eval = model(x, edge_index)
                    pred = model.predict_assign((h_eval, S_eval)).cpu().numpy()
            metrics = compute_all(y_np, pred)
            
            if metrics["ACC"] > best_acc:
                best_acc = metrics["ACC"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                logger.info(f"[BEST] Epoch {epoch}: ACC={metrics['ACC']:.4f} NMI={metrics['NMI']:.4f}")
        
        timing.append({"epoch": epoch, "duration": dur, "loss": float(loss.item()), **metrics})
        msg = f"Epoch {epoch}/{pretrain_epochs} loss={loss.item():.4f}"
        if comps:
            msg += " " + " ".join(f"{k}={v:.4f}" for k, v in comps.items())
        if metrics:
            msg += f" ACC={metrics['ACC']:.4f} NMI={metrics['NMI']:.4f}"
        logger.info(f"[TIME] pretrain_epoch {msg} took {dur:.2f}s")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model with ACC={best_acc:.4f}")

    # K-means initialization for MAGI centers
    if backbone_name == "magi":
        model.eval()
        with torch.no_grad():
            z = model(x, edge_index)
        logger.info("Initializing cluster centers with K-means...")
        kmeans_labels = model.init_centers_kmeans(z)
        
        # Fine-tune after K-means initialization
        logger.info("Fine-tuning after K-means initialization...")
        optimizer = optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-3) * 0.1, weight_decay=train_cfg.get("weight_decay", 5e-4))
        
        for epoch in range(1, 21):  # 20 epochs of fine-tuning
            model.train()
            optimizer.zero_grad()
            x1 = feature_dropout(x, feat_drop)
            x2 = feature_dropout(x, feat_drop)
            e1 = edge_dropout(edge_index, edge_drop)
            e2 = edge_dropout(edge_index, edge_drop)
            
            with maybe_autocast(device, train_cfg.get("amp", True)):
                z1 = model(x1, e1)
                z2 = model(x2, e2)
                loss, comps = model.loss_pretrain(z1, z2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    z_eval = model(x, edge_index)
                    pred = model.predict_assign(z_eval).cpu().numpy()
                metrics = compute_all(y_np, pred)
                logger.info(f"[FT] Epoch {epoch}: loss={loss.item():.4f} ACC={metrics['ACC']:.4f} NMI={metrics['NMI']:.4f}")

    # Final assignments and checkpoint
    model.eval()
    with torch.no_grad():
        x1 = feature_dropout(x, feat_drop)
        x2 = feature_dropout(x, feat_drop)
        e1 = edge_dropout(edge_index, edge_drop)
        e2 = edge_dropout(edge_index, edge_drop)
        
        if backbone_name == "magi":
            z1 = model(x1, e1)
            z2 = model(x2, e2)
            c1 = model.predict_assign(z1)
            c2 = model.predict_assign(z2)
            z_final = model(x, edge_index)
            pred_final = model.predict_assign(z_final).cpu().numpy()
        else:
            h1, S1 = model(x1, e1)
            h2, S2 = model(x2, e2)
            c1 = model.predict_assign((h1, S1))
            c2 = model.predict_assign((h2, S2))
            h_final, S_final = model(x, edge_index)
            pred_final = model.predict_assign((h_final, S_final)).cpu().numpy()
        
        S_set = (c1 != c2).nonzero(as_tuple=False).view(-1).cpu()
        logger.info(f"S size={S_set.numel()}")

    # Final metrics
    final_metrics = compute_all(y_np, pred_final)
    logger.info(f"[FINAL] ACC={final_metrics['ACC']:.4f} NMI={final_metrics['NMI']:.4f} ARI={final_metrics['ARI']:.4f} F1={final_metrics['F1']:.4f}")

    os.makedirs(run_dir, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "backbone": backbone_name,
        "assign_c1": c1.cpu(),
        "assign_c2": c2.cpu(),
        "S": S_set,
        "K": K_model,
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "proj_dim": proj_dim,
        "final_metrics": final_metrics,
    }
    ckpt_file = os.path.join(run_dir, "checkpoint.pt")
    torch.save(ckpt, ckpt_file)
    with open(os.path.join(run_dir, "timing_report.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    
    # Save final metrics
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Extra copy for downstream scripts
    ckpt_dir = train_cfg.get("checkpoint_path", "./experiments/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(ckpt_dir, f"{dataset}-{backbone_name}-pretrain.pt"))
    logger.info(f"Checkpoint saved to {ckpt_file}")


if __name__ == "__main__":
    main()
