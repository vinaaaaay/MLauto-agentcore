import json
import logging
import os
import re
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


# ── Code Extraction ──

def extract_code(response: str, language: str) -> str:
    """Extract a fenced code block from an LLM response."""
    if language == "python":
        pattern = r"```python\s*\n(.*?)```"
    elif language == "bash":
        pattern = r"```bash\s*\n(.*?)```"
    else:
        raise ValueError(f"Unsupported language: {language}")

    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    generic = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if generic:
        logger.warning(f"No {language} block found; using generic code block.")
        return generic[0].strip()

    logger.warning(f"No code block found; returning full response.")
    return response.strip()


# ── Requirements Resolver ──

def get_requirements_contents(registry_path: str, tool_name: str) -> Tuple[str, str]:
    """Reads requirements_common.txt and tool-specific requirements.txt from host registry path."""
    common_content = ""
    tool_content = ""
    if not registry_path:
        return common_content, tool_content
    
    reg_path = Path(registry_path)
    common_file = reg_path / "_common" / "requirements.txt"
    if common_file.exists():
        try:
            with open(common_file, "r") as f:
                common_content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read common requirements: {e}")
            
    catalog_file = reg_path / "_common" / "catalog.json"
    if catalog_file.exists() and tool_name:
        try:
            with open(catalog_file, "r") as f:
                catalog = json.load(f)
            tool_data = catalog.get("tools", {}).get(tool_name)
            if tool_data and "path" in tool_data:
                tool_req_file = reg_path / tool_data["path"] / "requirements.txt"
                if tool_req_file.exists():
                    with open(tool_req_file, "r") as f:
                        tool_content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read tool requirements for {tool_name}: {e}")
            
    return common_content, tool_content
