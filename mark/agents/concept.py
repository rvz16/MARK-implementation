import json
import os
from typing import Dict, List, Optional, Any

import torch

from mark.agents.client import LLMClient
from mark.agents.prompts import CONCEPT_PROMPT
from mark.utils import setup_logger, TokenCostTracker


def induce_concepts(
    H: torch.Tensor,
    assign: torch.Tensor,
    top_n: int,
    client: LLMClient,
    texts: Optional[List[str]] = None,
    run_dir: Optional[str] = None,
    tracker: Optional[TokenCostTracker] = None,
):
    """
    Concept Agent: Induce cluster concepts from top-n high-confidence samples.
    """
    logger = setup_logger()
    texts = texts or [f"node_{i}" for i in range(H.size(0))]
    clusters = assign.unique().tolist()
    centers = []
    
    for k in clusters:
        idx = (assign == k).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            centers.append(None)
            continue
        center = torch.nn.functional.normalize(H[idx].mean(dim=0, keepdim=True), p=2, dim=-1)
        centers.append(center)

    messages = []
    cluster_ids = []
    
    for ci, k in enumerate(clusters):
        idx = (assign == k).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0 or centers[ci] is None:
            continue
        h = torch.nn.functional.normalize(H[idx], p=2, dim=-1)
        sims = (h @ centers[ci].t()).view(-1)
        top_idx = sims.topk(min(top_n, idx.numel())).indices
        selected = idx[top_idx].tolist()
        phrases = [texts[j] for j in selected]
        user_content = "Cluster {} top nodes:\n{}".format(k, "\n".join(phrases[:20]))  # Limit to 20 for context
        messages.append([{"role": "system", "content": CONCEPT_PROMPT}, {"role": "user", "content": user_content}])
        cluster_ids.append(k)

    if not messages:
        return {}

    # Use synchronous batch_chat
    responses = client.batch_chat(messages)
    
    results: Dict[int, Dict[str, Any]] = {}
    os.makedirs(run_dir or ".", exist_ok=True)
    cache_path = os.path.join(run_dir, "concept.jsonl") if run_dir else None
    
    if cache_path:
        f = open(cache_path, "a", encoding="utf-8")
    else:
        f = None
    
    try:
        for k, (resp, usage) in zip(cluster_ids, responses):
            # Handle different response formats
            title = resp.get("cluster_title", "") or resp.get("title", "") or resp.get("name", "")
            keywords = resp.get("keywords", [])
            results[k] = {"title": title, "keywords": keywords}
            
            if f:
                f.write(json.dumps({"cluster_id": k, "response": resp, "usage": usage}) + "\n")
            logger.info(f"[Concept] cluster {k}: {results[k]}")
            
            if tracker and usage:
                tracker.add_usage(usage)
    finally:
        if f:
            f.close()
    
    return results
