import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

# Custom log level: DETAIL (between DEBUG=10 and INFO=20)
DETAIL_LEVEL = 15
logging.addLevelName(DETAIL_LEVEL, "DETAIL")

def detail(self, msg, *args, **kw):
    if self.isEnabledFor(DETAIL_LEVEL):
        kw.setdefault("stacklevel", 2)
        self._log(DETAIL_LEVEL, msg, args, **kw)

logging.Logger.detail = detail  # type: ignore

def configure_logging(output_dir: str, verbosity: int = 2) -> None:
    """
    Set up logging handlers for both console and file output.
    Args:
        output_dir: Directory where log files will be written.
        verbosity:
            0 = ERROR only
            1 = WARNING
            2 = INFO (default)
            3 = DETAIL
            4 = DEBUG (everything)
    """
    level_map = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: DETAIL_LEVEL,
        4: logging.DEBUG,
    }
    console_level = level_map.get(verbosity, logging.INFO)

    os.makedirs(output_dir, exist_ok=True)

    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    handlers.append(console_handler)

    # File: logs.txt (mirrors console)
    console_file = logging.FileHandler(
        os.path.join(output_dir, "logs.txt"), mode="w", encoding="utf-8"
    )
    console_file.setLevel(console_level)
    file_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_file.setFormatter(file_fmt)
    handlers.append(console_file)

    # File: debug_logs.txt (captures EVERYTHING)
    debug_file = logging.FileHandler(
        os.path.join(output_dir, "debug_logs.txt"), mode="w", encoding="utf-8"
    )
    debug_file.setLevel(logging.DEBUG)
    debug_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)s:%(funcName)s:%(lineno)d │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    debug_file.setFormatter(debug_fmt)
    handlers.append(debug_file)

    # Apply to root logger
    logging.basicConfig(
        level=logging.DEBUG,  # root captures everything; handlers filter
        handlers=handlers,
        force=True,
    )

class A2ACallLogger:
    """
    Logs every A2A agent call (request + response) to a structured JSONL file.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "a2a_calls.jsonl")
        self.logger = logging.getLogger("mlorchestrator.a2a")
        self.call_count = 0

    def _append_to_json_file(self, filename: str, record: dict) -> None:
        filepath = os.path.join(self.output_dir, filename)
        try:
            data = []
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = []
                    except Exception:
                        data = []
            data.append(record)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Failed to write to JSON log file {filepath}: {e}")

    def log_call(self, agent_name: str, skill: str, request_data: dict, response_data: dict, elapsed: float) -> None:
        self.call_count += 1
        call_id = self.call_count

        self.logger.info(
            f"[A2A Call #{call_id}] {agent_name} ({skill}) completed in {elapsed:.2f}s"
        )
        
        record = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "skill": skill,
            "elapsed_seconds": round(elapsed, 2),
            "request": request_data,
            "response": response_data,
        }
        
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to write A2A call log: {e}")

        # Map agents to JSON log filenames
        agent_file_map = {
            "PerceptionAgent": "perception_agent.json",
            "MCTSHandler": "mcts_handler.json",
            "MemoryAgent": "semantic_agent.json",
            "CodingAgent": "coder_agent.json"
        }

        if agent_name in agent_file_map:
            filename = agent_file_map[agent_name]
            log_record = {
                "timestamp": record["timestamp"],
                "skill": skill,
                "input": request_data,
                "output": response_data,
                "time_taken_seconds": round(elapsed, 3)
            }
            
            # For MemoryAgent, if mcp_calls are present in response, log them to mcp_server.json
            if agent_name == "MemoryAgent" and isinstance(response_data, dict) and "mcp_calls" in response_data:
                mcp_calls = response_data.get("mcp_calls", [])
                if mcp_calls:
                    for call in mcp_calls:
                        self._append_to_json_file("mcp_server.json", call)
                
                # Strip mcp_calls from logged response of semantic_agent.json to keep it clean
                log_record["output"] = {k: v for k, v in response_data.items() if k != "mcp_calls"}

            self._append_to_json_file(filename, log_record)
