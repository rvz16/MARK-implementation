import json
import os
from typing import Dict, Iterable, Set, Optional, Any

from mark.agents.client import LLMClient
from mark.agents.prompts import INFERENCE_PROMPT
from mark.utils import setup_logger, TokenCostTracker


def classify_consistency(
    S: Iterable[int],
    raw_texts: Dict[int, str],
    synth_texts: Dict[int, str],
    concepts: Dict[int, Dict[str, Any]],
    K: int,
    client: LLMClient,
    run_dir: Optional[str] = None,
    tracker: Optional[TokenCostTracker] = None,
):
    """
    Inference Agent: Classify nodes and filter by consistency between original and synthetic.
    """
    logger = setup_logger()
    S_list = list(S)
    
    if not S_list:
        return set(), {k: {"agree": 0, "total": 0} for k in range(K)}, {}
    
    messages_raw = []
    messages_syn = []
    
    concept_str = "\n".join(
        f"Cluster {cid}: {info.get('title', '')} (Keywords: {', '.join(info.get('keywords', []))})"
        for cid, info in concepts.items()
    )

    for i in S_list:
        orig = raw_texts.get(i, "")
        summ = synth_texts.get(i, "")
        user_raw = f"Original text:\n{orig}\nSynthetic summary:\n{summ}\n\nCluster Concepts:\n{concept_str}"
        user_syn = f"Original text:\n{summ}\nSynthetic summary:\n{summ}\n\nCluster Concepts:\n{concept_str}"
        messages_raw.append([{"role": "system", "content": INFERENCE_PROMPT}, {"role": "user", "content": user_raw}])
        messages_syn.append([{"role": "system", "content": INFERENCE_PROMPT}, {"role": "user", "content": user_syn}])

    # Use synchronous batch_chat
    resp_raw = client.batch_chat(messages_raw)
    resp_syn = client.batch_chat(messages_syn)

    R: Set[int] = set()
    stats: Dict[int, Dict[str, int]] = {k: {"agree": 0, "total": 0} for k in range(K)}
    pred_labels: Dict[int, int] = {}

    os.makedirs(run_dir or ".", exist_ok=True)
    cache_path = os.path.join(run_dir, "inference.jsonl") if run_dir else None
    cache_f = open(cache_path, "a", encoding="utf-8") if cache_path else None

    try:
        for node_id, raw_pack, syn_pack in zip(S_list, resp_raw, resp_syn):
            raw_resp, raw_usage = raw_pack
            syn_resp, syn_usage = syn_pack
            
            # Parse cluster_id from response
            try:
                cid_raw = int(raw_resp.get("cluster_id", -1))
            except (ValueError, TypeError):
                cid_raw = -1
            
            try:
                cid_syn = int(syn_resp.get("cluster_id", -1))
            except (ValueError, TypeError):
                cid_syn = -1
            
            if 0 <= cid_raw < K:
                stats[cid_raw]["total"] += 1
            
            # Consistency check: both classify to same cluster
            if cid_raw == cid_syn and 0 <= cid_raw < K:
                R.add(node_id)
                stats[cid_raw]["agree"] += 1
            
            if 0 <= cid_raw < K:
                pred_labels[node_id] = cid_raw
            
            if cache_f:
                cache_f.write(
                    json.dumps({
                        "node": node_id,
                        "raw": raw_resp,
                        "syn": syn_resp,
                        "raw_usage": raw_usage,
                        "syn_usage": syn_usage,
                    }) + "\n"
                )
            
            logger.info(f"[Inference] node {node_id}: raw={cid_raw} syn={cid_syn} agree={node_id in R}")
            
            if tracker:
                if raw_usage:
                    tracker.add_usage(raw_usage)
                if syn_usage:
                    tracker.add_usage(syn_usage)
    finally:
        if cache_f:
            cache_f.close()
    
    return R, stats, pred_labels
