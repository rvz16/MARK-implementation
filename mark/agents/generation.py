import json
import os
from typing import Dict, Iterable, List, Optional, Any

from mark.agents.client import LLMClient
from mark.agents.prompts import GENERATION_PROMPT
from mark.utils import setup_logger, TokenCostTracker


def synthesize_for_S(
    S: Iterable[int],
    neighbors: Dict[int, List[int]],
    texts: List[str],
    concepts: Dict[int, Dict[str, Any]],
    k: int,
    client: LLMClient,
    run_dir: Optional[str] = None,
    tracker: Optional[TokenCostTracker] = None,
) -> Dict[int, str]:
    """
    Generation Agent: Synthesize virtual text for uncertain nodes using neighbors.
    """
    logger = setup_logger()
    S_list = list(S)
    
    if not S_list:
        return {}
    
    messages = []
    
    concept_str = "\n".join(
        f"Cluster {cid}: {info.get('title', '')} (Keywords: {', '.join(info.get('keywords', []))})"
        for cid, info in concepts.items()
    )

    for i in S_list:
        neigh_ids = neighbors.get(i, [])[:k]
        neigh_texts = [texts[j] for j in neigh_ids if j < len(texts)]
        user_content = "Node: {}\nNeighbors:\n{}\n\nCluster Concepts:\n{}".format(
            texts[i] if i < len(texts) else f"node_{i}",
            "\n".join(f"- {t}" for t in neigh_texts) if neigh_texts else "- none",
            concept_str
        )
        messages.append([{"role": "system", "content": GENERATION_PROMPT}, {"role": "user", "content": user_content}])

    # Use synchronous batch_chat
    responses = client.batch_chat(messages)
    
    results: Dict[int, str] = {}
    os.makedirs(run_dir or ".", exist_ok=True)
    cache_path = os.path.join(run_dir, "generation.jsonl") if run_dir else None
    cache_f = open(cache_path, "a", encoding="utf-8") if cache_path else None
    
    try:
        for node_id, (resp, usage) in zip(S_list, responses):
            summary = resp.get("summary", "")
            results[node_id] = summary
            
            if cache_f:
                cache_f.write(json.dumps({"node": node_id, "response": resp, "usage": usage}) + "\n")
            logger.info(f"[Generation] node {node_id} summary len={len(summary)}")
            
            if tracker and usage:
                tracker.add_usage(usage)
    finally:
        if cache_f:
            cache_f.close()
    
    return results
