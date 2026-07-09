"""
Self-contained helper utilities for the Perception Agent.

Includes everything needed for standalone deployment:
  - LLM factory (_get_llm)
  - File system helpers (_get_all_files_sandbox, _group_similar_files, _pattern_to_path)
  - Code extraction & execution (_extract_code, _execute_code_sandbox)
  - Tool registry (_ToolRegistry)

No imports from MLauto, FAME, or any external agent modules.
Sandbox access is exclusively via AWS Lambda bastion gateway (SandboxClient).
"""

import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

try:
    from .sandbox_client import SandboxClient
except ImportError:
    from sandbox_client import SandboxClient

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

MAX_CHARS_PER_FILE = 768
MAX_FILE_GROUP_SIZE_TO_SHOW = 5
NUM_EXAMPLE_FILES_TO_SHOW = 1
DEFAULT_LIBRARY = "machine_learning"

_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "tools_registry"


# ═══════════════════════════════════════════════════════════════════════════
#  LLM Factory
# ═══════════════════════════════════════════════════════════════════════════

def _get_llm(config: dict = None):
    """
    Create and return a configured LLM instance.

    Supports OpenAI (gpt-*, o1-*, o3-*) and OpenRouter (any other model string).

    Args:
        config: Optional dict with keys: model, temperature, max_tokens.

    Returns:
        A ChatOpenAI or ChatOpenRouter instance ready for .invoke().
    """
    config = config or {}

    model = config.get("model", "gpt-4o-mini")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens") # None means use default API limit

    is_openai = (
        model.lower().startswith("gpt")
        or model.lower().startswith("o1-")
        or model.lower().startswith("o3-")
    )

    if is_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running: export OPENAI_API_KEY=sk-..."
            )

        is_openai_reasoning = any(x in model.lower() for x in ["o1-", "o3-", "gpt-5"])

        if is_openai_reasoning:
            logger.info("Detected OpenAI reasoning model. Forcing temp=1 and setting reasoning_effort='none'.")
            kwargs = {"model": model, "temperature": 1, "api_key": api_key, "max_retries": 1, "timeout": 60.0}
            kwargs["reasoning_effort"] = "none"
            if max_tokens is not None:
                kwargs["max_completion_tokens"] = max_tokens
            llm = ChatOpenAI(**kwargs)
        else:
            kwargs = {"model": model, "temperature": temperature, "api_key": api_key, "max_retries": 1, "timeout": 60.0}
            if "deepseek" in model.lower():
                kwargs["reasoning_effort"] = "none"
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            llm = ChatOpenAI(**kwargs)


        logger.info(f"Initialized OpenAI LLM: model={model}, temp={temperature}")
        return llm

    else:
        from langchain_openai import ChatOpenAI
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable is not set."
            )
        logger.info(f"Initialized OpenRouter via ChatOpenAI: model={model}, temp={temperature}")
        kwargs = {
            "model": model,
            "temperature": temperature,
            "api_key": openrouter_api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "max_retries": 1,
            "timeout": 60.0,
        }

        is_openai_reasoning = any(x in model.lower() for x in ["o1-", "o3-", "gpt-5"])
        if is_openai_reasoning:
            logger.info("Detected OpenAI reasoning model on OpenRouter. Forcing temp=1 and setting reasoning_effort='none'.")
            kwargs["temperature"] = 1
            kwargs["reasoning_effort"] = "none"
        elif "deepseek" in model.lower():
            logger.info("Detected DeepSeek model on OpenRouter. Setting reasoning_effort='none'.")
            kwargs["reasoning_effort"] = "none"

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════
#  Sandbox Client Factory
# ═══════════════════════════════════════════════════════════════════════════

def _get_sandbox_client(config: dict = None) -> SandboxClient:
    """Instantiate and return a SandboxClient using env vars."""
    sandbox_url = os.environ.get("SANDBOX_URL", "lambda:fame-sandbox-bastion")
    return SandboxClient(sandbox_url)


# ═══════════════════════════════════════════════════════════════════════════
#  File System Helpers (Sandbox-based)
# ═══════════════════════════════════════════════════════════════════════════

def _get_all_files_sandbox(
    folder_path: str,
    sandbox: SandboxClient,
    metric_logger: logging.Logger = None,
    ctx=None,
) -> list:
    """
    Recursively get all files in folder_path on the sandbox.

    Uses a compact tree structure bounded by directory count × extension count,
    NOT file count. This avoids blowing the Lambda response buffer on large datasets.

    Returns:
        list of (relative_path, absolute_path, size_bytes).
    """
    python_code = f"""
import os, json
folder = {repr(folder_path)}
tree = {{}}
total = 0
if os.path.exists(folder):
    for root, _, files in os.walk(folder):
        rel_root = os.path.relpath(root, folder)
        cur = tree
        if rel_root != '.':
            for p in rel_root.split(os.sep):
                cur = cur.setdefault(p, {{}})

        by_ext = {{}}
        for f in files:
            total += 1
            ext = os.path.splitext(f)[1].lower() or "NO_EXT"
            by_ext.setdefault(ext, []).append(f)

        for ext, names in by_ext.items():
            cnt = len(names)
            entry = {{"count": cnt}}
            file_entries = []
            for n in names[:5]:
                abs_n = os.path.join(root, n)
                try:
                    sz = os.path.getsize(abs_n)
                except Exception:
                    sz = 0
                file_entries.append((os.path.join(rel_root, n), abs_n, sz))
                if cnt > 5:
                    break
            entry["files"] = file_entries
            cur[ext] = entry

print(json.dumps({{"tree": tree, "total_count": total}}))
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
        result = json.loads(stdout)
        tree = result.get("tree", {})
        total_count = result.get("total_count", 0)
        files = _flatten_file_tree(tree)

        if total_count > 10000:
            logger.warning(
                f"  Folder {folder_path} has {total_count} files; returning {len(files)} samples"
            )

        return files
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing file list from sandbox: {e}. Output was: {stdout}")
        return []


def _flatten_file_tree(tree: dict) -> list:
    """Recursively flatten a compact file tree into a list of (rel, abs, size) tuples."""
    files = []
    for val in tree.values():
        if isinstance(val, dict):
            if "files" in val and isinstance(val["files"], list):
                files.extend(val["files"])
            else:
                files.extend(_flatten_file_tree(val))
    return files


def _group_similar_files(files: list) -> dict:
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


def _execute_code_sandbox(
    code: str,
    language: str,
    sandbox: SandboxClient,
    timeout: int = 60,
) -> tuple:
    """
    Execute code on the sandbox environment and return (success, stdout, stderr).
    """
    ext = "py" if language.lower() == "python" else "sh"
    temp_file = f"/tmp/sandbox_code_{uuid.uuid4().hex}.{ext}"

    write_success = sandbox.write_file_sync(temp_file, code)
    if not write_success:
        return False, "", "Failed to write temp code file to sandbox"

    if language.lower() == "python":
        success, stdout, stderr = sandbox.exec_shell_sync(
            f"timeout {timeout} python3 {temp_file}", cwd="/home/gem/workspace"
        )
    else:
        success, stdout, stderr = sandbox.exec_shell_sync(
            f"timeout {timeout} bash {temp_file}", cwd="/home/gem/workspace"
        )

    sandbox.exec_shell_sync(f"rm -f {temp_file}", cwd="")
    return success, stdout, stderr


# ═══════════════════════════════════════════════════════════════════════════
#  Tool Registry
# ═══════════════════════════════════════════════════════════════════════════

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

    def __init__(self, registry_path=None):
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

    def format_tools_info(self) -> str:
        """Format all tools for inclusion in the ToolSelector prompt."""
        lines = []
        for name, info in self.tools.items():
            lines.append(f"Library Name: {name}")
            lines.append(f"Version: v{info['version']}")
            lines.append(f"Description: {info['description']}")
            lines.append("")
        return "\n".join(lines)
