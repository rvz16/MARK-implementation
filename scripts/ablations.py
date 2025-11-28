import json
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from mark.agents import LLMClient, induce_concepts, synthesize_for_S, classify_consistency
from mark.backbones import MAGIEncoder, DMoNEncoder
from mark.datasets import load_tag
from mark.metrics import compute_all
from mark.utils import (
    get_device,
    load_config,
    load_plm_cache,
    save_plm_cache,
    set_seed,
    setup_logger,
    TokenCostTracker,
)
from mark.ranking import calibration_loss
from mark.augment import feature_dropout, edge_dropout


def compute_centers(z: torch.Tensor, assign: torch.Tensor, K: int) -> torch.Tensor:
    z_n = F.normalize(z, p=2, dim=-1)
    centers = []
    for k in range(K):
        idx = (assign == k)
        if idx.sum() > 0:
            centers.append(z_n[idx].mean(dim=0))
        else:
            centers.append(torch.zeros(z.size(1), device=z.device))
    centers = torch.stack(centers, dim=0)
    return F.normalize(centers, p=2, dim=-1)


def adjust_dim(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
    if vec.numel() >= target_dim:
        return vec[:target_dim]
    pad = torch.zeros(target_dim - vec.numel(), device=vec.device)
    return torch.cat([vec, pad], dim=0)


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
    run_dir = os.path.join(log_dir, dataset, f"ablations-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    x, edge_index, texts, y, K_true = load_tag(dataset, data_dir)
    K = engine_cfg.get("num_clusters", 0) or K_true
    in_dim = x.size(1)
    hidden_dim = engine_cfg.get("hidden_dim", 128)
    proj_dim = engine_cfg.get("proj_dim", 64)
    backbone_name = engine_cfg.get("backbone", "magi").lower()

    ckpt_dir = train_cfg.get("checkpoint_path", "./experiments/checkpoints")
    ckpt_file = os.path.join(ckpt_dir, f"{dataset}-{backbone_name}-pretrain.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location="cpu")

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
    plm_model = SentenceTransformer(data_cfg.get("plm_model", "sentence-transformers/all-MiniLM-L6-v2"), device=str(device))
    plm_cache_path = os.path.join(run_dir, "plm_cache.pt")
    plm_cache = load_plm_cache(plm_cache_path)

    scenarios = [
        ("full", False, False, False),
        ("w/o_concept", True, False, False),
        ("w/o_generation", False, True, False),
        ("w/o_inference", False, False, True),
    ]
    results = {}

    for name, skip_concept, skip_generation, skip_inference in scenarios:
        logger.info(f"Running scenario {name}")
        # fresh model and features
        if backbone_name == "magi":
            model = MAGIEncoder(in_dim, hidden_dim, proj_dim, K, engine_cfg.get("tau_align", 0.5), engine_cfg.get("lambda_clu", 1.0))
        else:
            model = DMoNEncoder(in_dim, hidden_dim, K)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)
        feat = x.clone().to(device)
        edge_idx = edge_index.to(device)

        # disagreement set
        with torch.no_grad():
            if backbone_name == "magi":
                c1 = model.predict_assign(model(feature_dropout(feat, 0.1), edge_dropout(edge_idx, 0.1)))
                c2 = model.predict_assign(model(feature_dropout(feat, 0.1), edge_dropout(edge_idx, 0.1)))
            else:
                h1, S1 = model(feature_dropout(feat, 0.1), edge_dropout(edge_idx, 0.1))
                h2, S2 = model(feature_dropout(feat, 0.1), edge_dropout(edge_idx, 0.1))
                c1 = model.predict_assign((h1, S1))
                c2 = model.predict_assign((h2, S2))
            S_set = (c1 != c2).nonzero(as_tuple=False).view(-1).tolist()

        pred_labels = {}
        if not skip_concept or not skip_generation or not skip_inference:
            model.eval()
        if not skip_concept:
            with torch.no_grad():
                if backbone_name == "magi":
                    H = model(feat, edge_idx)
                    assign = model.predict_assign(H)
                else:
                    H, S_prob = model(feat, edge_idx)
                    assign = model.predict_assign((H, S_prob))
            induce_concepts(H, assign, agents_cfg.get("top_n_concept", 50), llm, texts=texts, run_dir=run_dir, tracker=tracker)
        gen_map = {}
        if not skip_generation:
            neighbors = {}
            src, dst = edge_index
            for s, d in zip(src.tolist(), dst.tolist()):
                neighbors.setdefault(s, []).append(d)
                neighbors.setdefault(d, []).append(s)
            gen_map = synthesize_for_S(S_set, neighbors, texts, agents_cfg.get("k_neighbors", 10), llm, run_dir=run_dir, tracker=tracker)
        if not skip_inference and gen_map:
            R, stats, pred_labels = classify_consistency(S_set, {i: texts[i] for i in range(len(texts))}, gen_map, K, llm, run_dir=run_dir, tracker=tracker)
            if R:
                summaries = [gen_map[i] for i in R if gen_map.get(i)]
                missing = [t for t in summaries if t not in plm_cache]
                if missing:
                    embeds = plm_model.encode(missing, batch_size=data_cfg.get("batch_size_plm", 32), convert_to_numpy=True, show_progress_bar=False)
                    for txt, emb in zip(missing, embeds):
                        plm_cache[txt] = emb
                    save_plm_cache(plm_cache, plm_cache_path)
                for node in R:
                    if node in gen_map and gen_map[node] in plm_cache:
                        emb = torch.tensor(plm_cache[gen_map[node]], device=device, dtype=feat.dtype)
                        emb = adjust_dim(emb, feat.size(1))
                        feat[node] = (feat[node] + emb) / 2.0

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-3), weight_decay=train_cfg.get("weight_decay", 5e-4))
        optimizer.zero_grad()
        x1 = feature_dropout(feat, 0.1)
        x2 = feature_dropout(feat, 0.1)
        e1 = edge_dropout(edge_idx, 0.1)
        e2 = edge_dropout(edge_idx, 0.1)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and train_cfg.get("amp", True))):
            if backbone_name == "magi":
                z1 = model(x1, e1)
                z2 = model(x2, e2)
                Leng, _ = model.loss_pretrain(z1, z2)
                z_base = model(feat, edge_idx)
                assign_model = model.predict_assign(z_base)
                assign_all = assign_model.clone()
            else:
                h1, S1 = model(x1, e1)
                Leng, _ = model.loss_pretrain(S1, e1)
                h_base, S_base = model(feat, edge_idx)
                z_base = F.normalize(h_base, p=2, dim=-1)
                assign_model = model.predict_assign((h_base, S_base))
                assign_all = assign_model.clone()
            for n, lab in pred_labels.items():
                if 0 <= lab < K:
                    assign_all[n] = lab
            centers = compute_centers(z_base, assign_all, K)
            l_cal = calibration_loss(z_base, centers, assign_all, engine_cfg.get("t_rank", 0.1))
            loss = Leng + l_cal
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if backbone_name == "magi":
                z_eval = model(feat, edge_idx)
                assign_eval = model.predict_assign(z_eval).cpu()
            else:
                h_eval, S_eval = model(feat, edge_idx)
                assign_eval = model.predict_assign((h_eval, S_eval)).cpu()
        metrics = compute_all(y.numpy(), assign_eval.numpy())
        results[name] = metrics
        logger.info(f"{name}: {metrics}")

    with open(os.path.join(run_dir, "ablations.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    tracker.save(os.path.join(run_dir, "costs.json"))
    logger.info(f"Ablations saved to {run_dir}")


if __name__ == "__main__":
    main()
