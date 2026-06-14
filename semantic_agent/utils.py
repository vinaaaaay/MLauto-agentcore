import json
import logging
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  LLM Call Logger
# ═══════════════════════════════════════════════════════════════════════════

class _LLMCallLogger:
    """Logs every LLM call (prompt + response) to structured JSONL."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "llm_calls.jsonl")
        self.call_count = 0

    def call(self, llm, prompt: str, node_name: str = "unknown") -> str:
        self.call_count += 1
        call_id = self.call_count

        logger.info(f"[Call #{call_id}] {node_name} — sending prompt ({len(prompt)} chars)")

        start = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - start
        content = response.content

        logger.info(
            f"[Call #{call_id}] {node_name} — received response "
            f"({len(content)} chars, {elapsed:.1f}s)"
        )

        record = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "node": node_name,
            "prompt_length": len(prompt),
            "response_length": len(content),
            "elapsed_seconds": round(elapsed, 2),
            "prompt": prompt,
            "response": content,
        }
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write LLM call log: {e}")

        return content
