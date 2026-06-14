"""
Self-contained helper utilities for the Perception Agent.

Includes everything needed for standalone deployment:
  - LLM factory (_get_llm)
  - LLM call logger (_LLMCallLogger)
  - State snapshot logger (_log_state_snapshot)
  - File system helpers (_get_all_files, _group_similar_files, _pattern_to_path)
  - Code extraction & execution (_extract_code, _execute_code)
  - Tool registry (_ToolRegistry)

No imports from MLauto, FAME, or any external agent modules.
"""

import json
import logging
import os
import re
import select
import subprocess
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import httpx
from langchain_openai import ChatOpenAI
from .sandbox import BastionSandboxClient

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

MAX_CHARS_PER_FILE = 768
MAX_FILE_GROUP_SIZE_TO_SHOW = 5
NUM_EXAMPLE_FILES_TO_SHOW = 1
DEFAULT_LIBRARY = "machine learning"

_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "tools_registry"


# ═══════════════════════════════════════════════════════════════════════════
#  LLM Factory
# ═══════════════════════════════════════════════════════════════════════════

def _get_llm(config: dict = None):
    """
    Create and return a configured ChatOpenAI or ChatOpenRouter instance.

    Args:
        config: Optional dict with keys: model, temperature, max_tokens.
                Falls back to sensible defaults.

    Returns:
        A ChatOpenAI or ChatOpenRouter instance ready for .invoke() or .ainvoke().
    """
    config = config or {}

    model = config.get("model", "gpt-4o")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 16384)

    is_openai = model.lower().startswith("gpt") or model.lower().startswith("o1-") or model.lower().startswith("o3-")

    if is_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running: export OPENAI_API_KEY=sk-..."
            )

        # Reasoning models (o1, o3, gpt-5) have strict parameter rules
        is_reasoning_model = any(x in model.lower() for x in ["o1-", "o3-", "gpt-5"])

        if is_reasoning_model:
            logger.info("Detected reasoning model. Forcing temp=1 and using max_completion_tokens.")
            llm = ChatOpenAI(
                model=model,
                temperature=1,  # Must be 1
                max_completion_tokens=max_tokens,
                api_key=api_key,
            )
        else:
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )

        logger.info(f"Initialized OpenAI LLM: model={model}, temp={temperature}")
        return llm
    else:
        from langchain_openrouter import ChatOpenRouter
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is not set."
            )
        
        logger.info(f"Initialized OpenRouter LLM: model={model}, temp={temperature}")
        return ChatOpenRouter(
            model=model,
            temperature=temperature,
            api_key=openrouter_api_key,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  LLM Call Logger
# ═══════════════════════════════════════════════════════════════════════════

class _LLMCallLogger:
    """
    Logs every LLM call (prompt + response) to both:
      - The Python logger (at DEBUG level)
      - A structured JSONL file for post-run analysis
    """

    def __init__(self, output_dir: str, ctx=None, metric_logger: logging.Logger = None):
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "llm_calls.jsonl")
        self.logger = logging.getLogger("perception_agent.llm")
        self.call_count = 0
        self.ctx = ctx
        self.metric_logger = metric_logger

    def call(self, llm, prompt: str, node_name: str = "unknown") -> str:
        """
        Invoke the LLM, log the full prompt and response, and return the response text.
        """
        self.call_count += 1
        call_id = self.call_count

        self.logger.info(
            f"[Call #{call_id}] {node_name} — sending prompt ({len(prompt)} chars)"
        )
        self.logger.debug(
            f"[Call #{call_id}] {node_name} — PROMPT:\n"
            f"{'='*60}\n{prompt}\n{'='*60}"
        )

        start = time.time()
        
        invoke_config = {}
        if self.ctx and self.metric_logger:
            from common.logging_callback import SessionMetricsCallback
            invoke_config = {"callbacks": [SessionMetricsCallback(ctx=self.ctx, metric_logger=self.metric_logger)]}

        response = llm.invoke(prompt, config=invoke_config if invoke_config else None)
        elapsed = time.time() - start
        content = response.content

        self.logger.info(
            f"[Call #{call_id}] {node_name} — received response "
            f"({len(content)} chars, {elapsed:.1f}s)"
        )
        self.logger.debug(
            f"[Call #{call_id}] {node_name} — RESPONSE:\n"
            f"{'='*60}\n{content}\n{'='*60}"
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
            self.logger.warning(f"Failed to write LLM call log: {e}")

        return content


# ═══════════════════════════════════════════════════════════════════════════
#  State Snapshot Logger
# ═══════════════════════════════════════════════════════════════════════════

def _log_state_snapshot(state: dict, label: str, output_dir: str) -> None:
    """Save a snapshot of the current state dict to a JSON file."""
    _snap_logger = logging.getLogger("perception_agent.state")

    keys_with_values = [k for k, v in state.items() if v]
    _snap_logger.info(f"State snapshot [{label}]: keys with values = {keys_with_values}")

    snapshots_dir = os.path.join(output_dir, "state_snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%H%M%S")
    safe_label = label.replace(" ", "_").replace("/", "_")
    snapshot_path = os.path.join(snapshots_dir, f"{timestamp}_{safe_label}.json")

    serializable = {}
    for k, v in state.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            if isinstance(v, str) and len(v) > 2000:
                serializable[k] = v[:2000] + f"... [TRUNCATED, total {len(v)} chars]"
            else:
                serializable[k] = v
        else:
            serializable[k] = str(v)

    try:
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        _snap_logger.debug(f"State snapshot saved to {snapshot_path}")
    except Exception as e:
        _snap_logger.warning(f"Failed to save state snapshot: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  File System Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_sandbox_client(config: dict = None) -> BastionSandboxClient:
    """Instantiate and return the BastionSandboxClient."""
    config = config or {}
    sandbox_cfg = config.get("sandbox", {})
    return BastionSandboxClient(
        gateway_lambda_name=sandbox_cfg.get("gateway_lambda_name"),
        target_ip=sandbox_cfg.get("target_ip"),
        target_port=sandbox_cfg.get("target_port"),
        region_name=sandbox_cfg.get("region_name")
    )


def _get_all_files_sandbox(folder_path: str, sandbox: BastionSandboxClient) -> list[tuple[str, str, int]]:
    """
    Recursively get all files in folder_path on the sandbox.
    Returns list of (relative_path, absolute_path, size_bytes).
    """
    import json
    import uuid
    
    python_code = f"""
import os, json
folder = {repr(folder_path)}
res = []
if os.path.exists(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, folder)
            try:
                size = os.path.getsize(abs_path)
            except Exception:
                size = 0
            res.append((rel_path, abs_path, size))
print(json.dumps(res))
"""
    
    temp_file = f"/tmp/walk_{uuid.uuid4().hex}.py"
    write_success = sandbox.write_file_sync(temp_file, python_code)
    if not write_success:
        logger.error("Failed to write temporary walk script to sandbox")
        return []
        
    success, stdout, stderr = sandbox.exec_shell_sync(f"python3 {temp_file}", cwd="")
    sandbox.exec_shell_sync(f"rm -f {temp_file}", cwd="")
    
    if not success:
        logger.error(f"Failed to list files on sandbox in {folder_path}: {stderr}")
        return []
    
    try:
        start_idx = stdout.find("[")
        end_idx = stdout.rfind("]")
        if start_idx != -1 and end_idx != -1:
            json_str = stdout[start_idx:end_idx+1]
            return json.loads(json_str)
        else:
            logger.error(f"Invalid output format when listing files: {stdout}")
            return []
    except Exception as e:
        logger.error(f"Error parsing file list from sandbox: {e}. Output was: {stdout}")
        return []


def _get_all_files(folder_path: str) -> list[tuple[str, str]]:
    """
    Recursively get all files in folder_path.

    Returns:
        List of (relative_path, absolute_path) tuples.
    """
    all_files = []
    abs_folder_path = os.path.abspath(folder_path)

    for root, _, files in os.walk(abs_folder_path):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, abs_folder_path)
            all_files.append((rel_path, abs_path))

    return all_files


def _group_similar_files(files: list[tuple[str, str]]) -> dict:
    """
    Group files by folder structure and extension.

    At each depth level, if there are ≤5 unique folders the actual names are
    used; otherwise a wildcard '*' is substituted.

    Returns:
        Dict mapping group-key tuples to lists of (rel_path, abs_path).
    """
    depth_folders: dict = defaultdict(set)

    for rel_path, _ in files:
        parts = os.path.normpath(rel_path).split(os.sep)
        for depth, folder in enumerate(parts[:-1]):
            depth_folders[depth].add(folder)

    groups: dict = defaultdict(list)
    for rel_path, abs_path in files:
        parts = os.path.normpath(rel_path).split(os.sep)
        folders = parts[:-1]
        filename = parts[-1]
        ext = os.path.splitext(filename)[1].lower()

        group_key_parts = []
        for depth, folder in enumerate(folders):
            if len(depth_folders[depth]) <= 5:
                group_key_parts.append(folder)
            else:
                group_key_parts.append("*")
        group_key_parts.append(ext if ext else "NO_EXT")

        groups[tuple(group_key_parts)].append((rel_path, abs_path))

    return groups


def _pattern_to_path(pattern: tuple, base_path: str) -> str:
    """Convert a group pattern tuple to a display path string."""
    folders = pattern[:-1]
    ext = pattern[-1]

    path_parts = list(str(f) for f in folders)
    path_parts.append("*" if ext == "NO_EXT" else f"*{ext}")

    relative_pattern = os.path.join(*path_parts) if path_parts else "*"
    return os.path.join(base_path, relative_pattern)


# ═══════════════════════════════════════════════════════════════════════════
#  Code Extraction & Execution
# ═══════════════════════════════════════════════════════════════════════════

def _extract_code(response: str, language: str) -> str:
    """
    Extract a fenced code block from an LLM response.

    Tries ```python or ```bash first, then generic ```, then full response.
    """
    if language == "python":
        pattern = r"```python\s*\n(.*?)```"
    elif language == "bash":
        pattern = r"```bash\s*\n(.*?)```"
    else:
        raise ValueError(f"Unsupported language: {language}")

    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Fallback: generic code block
    generic = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if generic:
        logger.warning(f"No {language} block found; using generic code block.")
        return generic[0].strip()

    logger.warning("No code block found; returning full response.")
    return response


def _execute_code_sandbox(code: str, language: str, sandbox: BastionSandboxClient, timeout: int = 3600) -> tuple[bool, str, str]:
    """
    Execute code on the sandbox environment and return (success, stdout, stderr).
    """
    import uuid
    temp_file = f"/tmp/sandbox_code_{uuid.uuid4().hex}.py" if language.lower() == "python" else f"/tmp/sandbox_code_{uuid.uuid4().hex}.sh"
    
    # Write code to sandbox
    write_success = sandbox.write_file_sync(temp_file, code)
    if not write_success:
        return False, "", "Failed to write temp code file to sandbox"
        
    # Execute code on sandbox
    if language.lower() == "python":
        success, stdout, stderr = sandbox.exec_shell_sync(f"python3 {temp_file}", cwd="/home/gem/workspace")
    else:
        success, stdout, stderr = sandbox.exec_shell_sync(f"bash {temp_file}", cwd="/home/gem/workspace")
        
    # Clean up temp file
    sandbox.exec_shell_sync(f"rm -f {temp_file}", cwd="")
    
    return success, stdout, stderr


def _execute_code(code: str, language: str, timeout: int = 3600) -> tuple[bool, str, str]:
    """
    Execute code with real-time output streaming and timeout.

    Args:
        code: Code string to execute.
        language: "python" or "bash".
        timeout: Maximum seconds before killing the process.

    Returns:
        (success, stdout, stderr)
    """
    if language.lower() == "python":
        cmd = ["python", "-c", code]
    elif language.lower() == "bash":
        cmd = ["bash", "-c", code]
    else:
        return False, "", f"Unsupported language: {language}"

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_chunks, stderr_chunks = [], []
        recent_stdout: deque = deque(maxlen=100)
        recent_stderr: deque = deque(maxlen=100)
        streams = [process.stdout, process.stderr]
        start_time = time.time()

        while streams:
            elapsed = time.time() - start_time
            remaining = max(0, timeout - elapsed)

            if remaining <= 0:
                process.terminate()
                time.sleep(3)
                if process.poll() is None:
                    process.kill()
                stdout_chunks.append(f"\nProcess reached time limit after {timeout} seconds.\n")
                logger.info(f"Process reached time limit after {timeout}s.")
                break

            readable, _, _ = select.select(streams, [], [], min(1, remaining))

            if not readable and process.poll() is None:
                continue
            if not readable and process.poll() is not None:
                break

            for stream in readable:
                line = stream.readline()
                if not line:
                    streams.remove(stream)
                    continue

                if stream == process.stdout:
                    if line not in recent_stdout:
                        recent_stdout.append(line)
                        stdout_chunks.append(line)
                else:
                    if line not in recent_stderr:
                        recent_stderr.append(line)
                        stderr_chunks.append(line)

        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                stderr_chunks.append("Process forcibly terminated after timeout\n")

        success = process.returncode == 0
        return success, "".join(stdout_chunks), "".join(stderr_chunks)

    except Exception as e:
        return False, "", f"Error executing {language} code: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
#  Tool Registry
# ═══════════════════════════════════════════════════════════════════════════

class TutorialInfo(NamedTuple):
    """Stores information about a tutorial."""
    path: Path
    title: str
    summary: str
    score: Optional[float] = None
    content: Optional[str] = None


class _ToolRegistry:
    """
    Reads the tool catalog and per-tool metadata from disk.

    Uses the bundled tools_registry/ directory by default,
    making this agent fully self-contained.

    Usage:
        registry = _ToolRegistry()
        tools = registry.list_tools()
        info  = registry.get_tool("autogluon.tabular")
    """

    def __init__(self, registry_path: str | Path | None = None):
        self.registry_path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY_PATH
        self.catalog_path = self.registry_path / "_common" / "catalog.json"
        self._cache: Optional[dict] = None

    @property
    def tools(self) -> dict:
        if self._cache is None:
            self._load()
        return self._cache

    def _load(self) -> None:
        """Load catalog.json and merge each tool's tool.json."""
        try:
            with open(self.catalog_path, "r") as f:
                catalog = json.load(f)
        except FileNotFoundError:
            logger.warning(f"catalog.json not found at {self.catalog_path}. Using empty registry.")
            self._cache = {}
            return

        tools_info = {}
        for tool_name, tool_data in catalog.get("tools", {}).items():
            tool_dir = self.registry_path / tool_data["path"]
            tool_json_path = tool_dir / "tool.json"

            info = {
                "name": tool_name,
                "path": tool_data["path"],
                "version": tool_data.get("version", "0.0.0"),
                "description": tool_data.get("description", ""),
                "requirements": [],
                "prompt_template": [],
            }

            if tool_json_path.exists():
                try:
                    with open(tool_json_path, "r") as f:
                        tj = json.load(f)
                    info["requirements"] = tj.get("requirements", [])
                    info["prompt_template"] = tj.get("prompt_template", [])
                except Exception as e:
                    logger.warning(f"Error loading tool.json for {tool_name}: {e}")

            req_path = tool_dir / "requirements.txt"
            if req_path.exists():
                try:
                    info["requirements"] = [
                        line.strip() for line in req_path.read_text().splitlines() if line.strip()
                    ]
                except Exception as e:
                    logger.warning(f"Error loading requirements.txt for {tool_name}: {e}")

            tools_info[tool_name] = info

        self._cache = tools_info

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool(self, name: str) -> Optional[dict]:
        return self.tools.get(name)

    def get_tool_prompt(self, name: str) -> str:
        """Return the prompt_template for a tool as a single string."""
        tool = self.get_tool(name)
        if not tool:
            return ""
        pt = tool.get("prompt_template", [])
        if isinstance(pt, list):
            return "\n".join(pt)
        return str(pt)

    def get_tool_path(self, name: str) -> Optional[Path]:
        """Get the absolute path for a tool's directory."""
        tool = self.get_tool(name)
        if not tool:
            return None
        return self.registry_path / tool["path"]

    def get_tool_tutorials_folder(self, name: str, condensed: bool = False) -> Path:
        """Get the tutorials folder for a specific tool."""
        tool_path = self.get_tool_path(name)
        if tool_path is None:
            raise FileNotFoundError(f"Tool {name} not found in registry")
        subfolder = "condensed_tutorials" if condensed else "tutorials"
        tutorials_dir = tool_path / subfolder
        if not tutorials_dir.exists():
            raise FileNotFoundError(f"No {subfolder} found for tool {name} at {tutorials_dir}")
        return tutorials_dir

    def get_common_requirements_file(self) -> Path:
        """Get the path to _common/requirements.txt."""
        return self.registry_path / "_common" / "requirements.txt"

    def get_tool_requirements_file(self, name: str) -> Path:
        """Get the path to a tool's requirements.txt."""
        tool_path = self.get_tool_path(name)
        if tool_path is None:
            raise FileNotFoundError(f"Tool {name} not found")
        return tool_path / "requirements.txt"

    def format_tools_info(self) -> str:
        """Format all tools for inclusion in the ToolSelector prompt."""
        lines = []
        for name, info in self.tools.items():
            lines.append(f"Library Name: {name}")
            lines.append(f"Version: v{info['version']}")
            lines.append(f"Description: {info['description']}")
            lines.append("")
        return "\n".join(lines)
