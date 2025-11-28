import json
import time
from typing import Any, Dict, List, Tuple, Optional
import concurrent.futures

from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 256,
        concurrency: int = 2,
        retries: int = 3,
        retry_backoff: float = 1.5,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.concurrency = concurrency
        self.retries = retries
        self.retry_backoff = retry_backoff

    def _parse_json(self, content: str) -> Dict[str, Any]:
        if not content:
            return {}
        
        content = content.strip()
        
        # First try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to repair truncated JSON by adding closing brace/quote
        for suffix in ['"}', '"}', '" }', '"}', '}', '"}']:
            try:
                return json.loads(content + suffix)
            except json.JSONDecodeError:
                continue
        
        # Try to extract key-value pairs manually
        import re
        result = {}
        
        # Extract summary field
        summary_match = re.search(r'"summary"\s*:\s*"([^"]*(?:"[^"]*)*)', content)
        if summary_match:
            result["summary"] = summary_match.group(1).rstrip('"').rstrip()
        
        # Extract cluster_title field
        title_match = re.search(r'"(?:cluster_title|title)"\s*:\s*"([^"]*)"', content)
        if title_match:
            result["title"] = title_match.group(1)
        
        # Extract cluster_id field
        cid_match = re.search(r'"cluster_id"\s*:\s*(\d+)', content)
        if cid_match:
            result["cluster_id"] = int(cid_match.group(1))
        
        # Extract keywords field
        kw_match = re.search(r'"keywords"\s*:\s*\[(.*?)\]', content, re.DOTALL)
        if kw_match:
            kw_str = kw_match.group(1)
            keywords = re.findall(r'"([^"]+)"', kw_str)
            result["keywords"] = keywords
        
        if result:
            return result
        
        print(f"[LLM] Failed to parse JSON: {content[:200]}")
        return {}

    def chat(self, messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Synchronous single chat request."""
        for attempt in range(self.retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                usage = resp.usage.to_dict() if hasattr(resp.usage, "to_dict") else dict(resp.usage) if resp.usage else {}
                return self._parse_json(content), usage
            except Exception as e:
                if attempt == self.retries - 1:
                    print(f"[LLM] Error after {self.retries} retries: {e}")
                    return {}, None
                time.sleep(self.retry_backoff ** attempt)
        return {}, None

    def batch_chat(
        self, 
        messages_list: List[List[Dict[str, str]]]
    ) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
        """
        Synchronous batch chat using ThreadPoolExecutor for concurrency.
        This avoids asyncio issues with nested event loops.
        """
        if not messages_list:
            return []
        
        results = [None] * len(messages_list)
        
        # Use ThreadPoolExecutor for simple parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {executor.submit(self.chat, msgs): i for i, msgs in enumerate(messages_list)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[LLM] Request {idx} failed: {e}")
                    results[idx] = ({}, None)
        
        return results
