import os
import time
import json
from datetime import datetime
from typing import Dict, List, Set

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from mark.augment import feature_dropout, edge_dropout
from mark.backbones import MAGIEncoder, DMoNEncoder
from mark.datasets import load_tag
from mark.metrics import compute_all, save_metrics
from mark.utils import (
    set_seed,
    load_config,
    setup_logger,
    get_device,
    maybe_autocast,
    save_config_snapshot,
    load_plm_cache,
    save_plm_cache,
    TokenCostTracker,
)
from mark.agents import LLMClient, induce_concepts, synthesize_for_S, classify_consistency
from mark.ranking import calibration_loss, ranking_calibration_loss


def make_run_dir(base: str, dataset: str, prefix: str) -> str:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(base, dataset, f"{prefix}-{run_id}")
    os.makedirs(path, exist_ok=True)
    return path


def build_neighbors(edge_index: torch.Tensor, num_nodes: int) -> Dict[int, List[int]]:
    """Build adjacency list from edge index."""
    neighbors = {i: [] for i in range(num_nodes)}
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        neighbors[s].append(d)
        neighbors[d].append(s)
    # Remove duplicates
    for i in neighbors:
        neighbors[i] = list(set(neighbors[i]))
    return neighbors


def compute_centers(z: torch.Tensor, assign: torch.Tensor, K: int) -> torch.Tensor:
    """Compute cluster centers from embeddings and assignments."""
    centers = []
    z_n = F.normalize(z, p=2, dim=-1)
    for k in range(K):
        idx = (assign == k)
        if idx.sum() > 0:
            centers.append(z_n[idx].mean(dim=0))
        else:
            centers.append(torch.zeros(z.size(1), device=z.device))
    centers = torch.stack(centers, dim=0)
    centers = F.normalize(centers, p=2, dim=-1)
    return centers


def adjust_dim(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Adjust vector dimension to match target."""
    if vec.numel() >= target_dim:
        return vec[:target_dim]
    pad = torch.zeros(target_dim - vec.numel(), device=vec.device)
    return torch.cat([vec, pad], dim=0)


def select_uncertain_nodes(
    model, 
    x: torch.Tensor, 
    edge_index: torch.Tensor, 
    backbone_name: str,
    feat_drop: float = 0.1,
    edge_drop: float = 0.1,
) -> torch.Tensor:
    """
    Select uncertain nodes based on disagreement between two augmented views.
    Nodes with different cluster assignments across views are considered uncertain.
    """
    model.eval()
    with torch.no_grad():
        x_a = feature_dropout(x, feat_drop)
        e_a = edge_dropout(edge_index, edge_drop)
        x_b = feature_dropout(x, feat_drop)
        e_b = edge_dropout(edge_index, edge_drop)
        
        if backbone_name == "magi":
            c1 = model.predict_assign(model(x_a, e_a))
            c2 = model.predict_assign(model(x_b, e_b))
        else:
            h1, S1 = model(x_a, e_a)
            h2, S2 = model(x_b, e_b)
            c1 = model.predict_assign((h1, S1))
            c2 = model.predict_assign((h2, S2))
        
        S_set = (c1 != c2).nonzero(as_tuple=False).view(-1)
    
    return S_set


def main():
    data_cfg = load_config("configs/data.yaml")
    engine_cfg = load_config("configs/engine.yaml")
    agents_cfg = load_config("configs/agents.yaml")
    train_cfg = load_config("configs/train.yaml")
    set_seed(train_cfg.get("seed", 42))
    device = get_device(train_cfg.get("device", "cpu"))
    logger = setup_logger()

    dataset = data_cfg.get("dataset", "cora")
    data_dir = data_cfg.get("data_dir", "./data")
    log_dir = train_cfg.get("log_dir", "./experiments")
    run_dir = make_run_dir(log_dir, dataset, "mark")
    save_config_snapshot({"data": data_cfg, "engine": engine_cfg, "agents": agents_cfg, "train": train_cfg}, os.path.join(run_dir, "config.yaml"))

    x, edge_index, texts, y, K_true = load_tag(dataset, data_dir)
    K = engine_cfg.get("num_clusters", 0) or K_true
    in_dim = x.size(1)
    hidden_dim = engine_cfg.get("hidden_dim", 256)
    proj_dim = engine_cfg.get("proj_dim", 128)
    backbone_name = engine_cfg.get("backbone", "magi").lower()

    ckpt_dir = train_cfg.get("checkpoint_path", "./experiments/checkpoints")
    ckpt_file = os.path.join(ckpt_dir, f"{dataset}-{backbone_name}-pretrain.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}. Run pretrain.py first.")
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # Load model with correct architecture
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

    x = x.to(device)
    edge_index = edge_index.to(device)
    y_np = y.numpy()
    neighbors = build_neighbors(edge_index.cpu(), x.size(0))

    # Initialize LLM client
    llm = LLMClient(
        base_url=agents_cfg["llm"]["base_url"],
        model=agents_cfg["llm"]["model"],
        temperature=agents_cfg["llm"].get("temperature", 0.2),
        max_tokens=agents_cfg["llm"].get("max_tokens", 256),
        concurrency=agents_cfg.get("concurrency", 2),
        retries=agents_cfg.get("retries", 3),
        retry_backoff=agents_cfg.get("retry_backoff", 1.5),
    )
    tracker = TokenCostTracker()
    plm_cache_path = os.path.join(run_dir, "plm_cache.pt")
    plm_cache = load_plm_cache(plm_cache_path)
    plm_model = None

    # Training setup
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=train_cfg.get("lr", 1e-3), 
        weight_decay=train_cfg.get("weight_decay", 5e-4)
    )
    
    T = train_cfg.get("T", 3)
    T_prime = train_cfg.get("T_prime", 1)
    ft_epochs = train_cfg.get("ft_epochs_per_step", 10)  # Increased
    t_rank = engine_cfg.get("t_rank", 0.1)
    feat_drop = engine_cfg.get("feat_drop", 0.1)
    edge_drop = engine_cfg.get("edge_drop", 0.1)
    
    timing = []
    all_R: Set[int] = set()
    all_pred_labels: Dict[int, int] = {}

    logger.info(f"Starting MARK training for {T} steps on {dataset}")
    logger.info(f"Model: {backbone_name}, K={K}, ft_epochs={ft_epochs}")

    for step in range(1, T + 1):
        step_start = time.perf_counter()
        
        # Select uncertain nodes via disagreement
        S_set = select_uncertain_nodes(model, x, edge_index, backbone_name, feat_drop, edge_drop)
        logger.info(f"[Step {step}] Found {len(S_set)} uncertain nodes")

        pred_labels: Dict[int, int] = {}
        R: Set[int] = set()
        
        if step % T_prime == 0 and len(S_set) > 0:
            logger.info(f"[Step {step}] Running multi-agent collaboration")
            model.eval()
            
            with torch.no_grad():
                if backbone_name == "magi":
                    H = model(x, edge_index)
                    assign = model.predict_assign(H)
                else:
                    H, S_prob = model(x, edge_index)
                    assign = model.predict_assign((H, S_prob))
            
            # Agent 1: Concept Induction
            logger.info("[Agent 1] Concept induction...")
            concept_info = induce_concepts(
                H, assign, 
                agents_cfg.get("top_n_concept", 50), 
                llm, 
                texts=texts, 
                run_dir=run_dir, 
                tracker=tracker
            )
            
            # Agent 2: Generation (synthesis)
            logger.info("[Agent 2] Text synthesis...")
            gen_map = synthesize_for_S(
                S_set.tolist(), 
                neighbors, 
                texts, 
                concept_info, 
                agents_cfg.get("k_neighbors", 10), 
                llm, 
                run_dir=run_dir, 
                tracker=tracker
            )
            
            # Agent 3: Inference and consistency filtering
            logger.info("[Agent 3] Inference and filtering...")
            R, stats, pred_labels = classify_consistency(
                S_set.tolist(), 
                {i: texts[i] for i in range(len(texts))}, 
                gen_map, 
                concept_info, 
                K, 
                llm, 
                run_dir=run_dir, 
                tracker=tracker
            )
            
            logger.info(f"[Step {step}] R size: {len(R)} (from S={len(S_set)})")
            
            # Update features for nodes in R using PLM embeddings
            if len(R) > 0:
                if plm_model is None:
                    plm_model = SentenceTransformer(
                        data_cfg.get("plm_model", "sentence-transformers/all-MiniLM-L6-v2"), 
                        device=str(device)
                    )
                
                new_texts = [gen_map[i] for i in R if gen_map.get(i)]
                missing = [t for t in new_texts if t not in plm_cache]
                
                if missing:
                    embeds = plm_model.encode(
                        missing, 
                        batch_size=data_cfg.get("batch_size_plm", 32), 
                        convert_to_numpy=True, 
                        show_progress_bar=False
                    )
                    for txt, emb in zip(missing, embeds):
                        plm_cache[txt] = emb
                    save_plm_cache(plm_cache, plm_cache_path)
                
                # Update node features with averaged PLM embeddings
                for node in R:
                    summary = gen_map.get(node, "")
                    if summary in plm_cache:
                        emb = torch.tensor(plm_cache[summary], device=device, dtype=x.dtype)
                        emb = adjust_dim(emb, x.size(1))
                        # Weighted average: 70% original, 30% new
                        x[node] = 0.7 * x[node] + 0.3 * emb
            
            # Accumulate predictions across steps
            all_R.update(R)
            all_pred_labels.update(pred_labels)

        # Fine-tuning phase
        model.train()
        best_loss = float('inf')
        
        for e in range(1, ft_epochs + 1):
            ft_start = time.perf_counter()
            optimizer.zero_grad()
            
            x1 = feature_dropout(x, feat_drop)
            x2 = feature_dropout(x, feat_drop)
            e1 = edge_dropout(edge_index, edge_drop)
            e2 = edge_dropout(edge_index, edge_drop)
            
            with maybe_autocast(device, train_cfg.get("amp", True)):
                if backbone_name == "magi":
                    z1 = model(x1, e1)
                    z2 = model(x2, e2)
                    Leng, comps = model.loss_pretrain(z1, z2)
                    z_base = model(x, edge_index)
                    assign_model = model.predict_assign(z_base)
                else:
                    h1, S1 = model(x1, e1)
                    Leng, comps = model.loss_pretrain(S1, e1)
                    h_base, S_base = model(x, edge_index)
                    z_base = F.normalize(h_base, p=2, dim=-1)
                    assign_model = model.predict_assign((h_base, S_base))
                
                # Create assignment with LLM predictions for nodes in R
                assign_all = assign_model.clone()
                for n, lab in all_pred_labels.items():
                    if 0 <= lab < K:
                        assign_all[n] = lab
                
                # Compute centers from current embeddings
                centers = compute_centers(z_base, assign_all, K)
                
                # Calibration loss only on reliable nodes
                if len(all_R) > 0:
                    l_cal = calibration_loss(z_base, centers, assign_all, t_rank, all_R)
                else:
                    l_cal = torch.tensor(0.0, device=device)
                
                loss = Leng + l_cal
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            ft_dur = time.perf_counter() - ft_start
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            if e % 2 == 0 or e == ft_epochs:
                logger.info(f"[FT] step={step} epoch={e} loss={loss.item():.4f} Leng={Leng.item():.4f} Lcal={l_cal.item() if isinstance(l_cal, torch.Tensor) else l_cal:.4f} took {ft_dur:.2f}s")
        
        step_dur = time.perf_counter() - step_start
        
        # Evaluate metrics
        model.eval()
        with torch.no_grad():
            if backbone_name == "magi":
                z_eval = model(x, edge_index)
                assign_eval = model.predict_assign(z_eval).cpu()
            else:
                h_eval, S_eval = model(x, edge_index)
                assign_eval = model.predict_assign((h_eval, S_eval)).cpu()
        
        metrics = compute_all(y_np, assign_eval.numpy())
        metrics_path_json = os.path.join(run_dir, f"metrics_step{step}.json")
        metrics_path_csv = os.path.join(run_dir, f"metrics_step{step}.csv")
        save_metrics(metrics, metrics_path_json, metrics_path_csv)
        
        logger.info(f"[Step {step}] Metrics: ACC={metrics['ACC']:.4f} NMI={metrics['NMI']:.4f} ARI={metrics['ARI']:.4f} F1={metrics['F1']:.4f}")
        
        timing.append({
            "step": step, 
            "duration": step_dur, 
            "R": len(R), 
            "S": len(S_set), 
            "total_R": len(all_R),
            **metrics
        })
        
        with open(os.path.join(run_dir, "timing_report.json"), "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)
        tracker.save(os.path.join(run_dir, "costs.json"))
        torch.save({"model_state": model.state_dict(), "step": step}, os.path.join(run_dir, f"checkpoint_step{step}.pt"))

    # Save final metrics
    save_metrics(metrics, os.path.join(run_dir, "metrics.json"), os.path.join(run_dir, "metrics.csv"))
    
    logger.info(f"Run finished at {run_dir}")
    logger.info(f"Final metrics: ACC={metrics['ACC']:.4f} NMI={metrics['NMI']:.4f} ARI={metrics['ARI']:.4f} F1={metrics['F1']:.4f}")


if __name__ == "__main__":
    main()
