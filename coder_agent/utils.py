import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

logger = logging.getLogger(__name__)

# ─── CoderAgent State Definition ───

class CoderAgentState(TypedDict, total=False):
    # Context Inputs
    task_description: str
    data_prompt: str
    user_input: str
    current_tool: str
    tool_prompt: str
    tutorial_prompt: str
    all_error_analyses: List[str]

    # Run Configuration
    config: Dict[str, Any]
    output_folder: str
    sandbox_client: Any

    # Current iteration tracking
    iteration: int
    node_id: int
    stage: str  # "root", "evolve", or "debug"

    # Previous attempts (if improving/debugging)
    previous_python_code: str
    previous_bash_script: str

    # Outputs
    python_code: str
    python_file_path: str
    bash_script: str
    stdout: str
    stderr: str
    decision: str  # "SUCCESS" or "FIX"
    error_summary: Optional[str]
    validation_score: Optional[float]
    error_message: str
    job_id: str   # Background execution job identifier (e.g. "node_1")

# ─── CoderAgent Prompts (Python) ───

PYTHON_CODER_PROMPT = """\
As an AutoML Agent, you will be given a folder containing data and description files. Please generate Python code using {current_tool} to train a predictor and make predictions on test data. Follow these specifications:

ONLY save files to the working directory: {output_folder}.

1. Data preprocessing:
   - Remove training data samples without valid labels (drop rows where the **label/target** is NA). 
   - **IMPORTANT**: Do NOT drop rows based on missing feature values, and do NOT perform manual missing value imputation. AutoGluon handles missing features automatically.
   - Remove the unnecessary index column (if applicable)

2. Model training:
   - Use {current_tool} with appropriate parameters for the task
   - If a model is trained, save it in a folder with random timestamp within {output_folder}

3. Prediction:
   - Make predictions on the ENTIRE test set, preserving ORIGINAL INDICES to maintain exact row correspondence. NEVER drop any test rows for any reason (including missing values), and ensure the output has the exact same number of rows as the test set.
   - Save the predicted results to {output_folder}, result file name should be "results", the format and extension should be same as the test data file
   - Output column names must exactly match those in the training or sample submission files without adding "predicted_" prefixes or creating any new columns.
   - IMPORTANT: At the end, implement validation checks that assert the prediction file maintains exact test data indices, verify correct column names match requirements, confirm proper output format, verify the number of predictions equals the number of test samples, and if applicable, sanity check output predictions are valid and correct.

4. Documentation:
   - Add a brief docstring at the beginning of the script explaining its purpose
   - Include additional installation steps with comments at the beginning of the script
   - Include comments explaining any complex operations or design decisions

5. Others:
   - To avoid DDP errors, wrap the code in: if __name__ == "__main__":
   - Ensure errors are propagated up and not silently caught - do not use try/except blocks unless you explicitly re-raise the exception.
   - **CRITICAL**: Do NOT use `n_jobs=-1` or parallel/multiprocess execution (e.g. in scikit-learn or joblib). Always run single-threaded (set `n_jobs=1` or omit it) to prevent CPU lockups/deadlocks in the sandbox container.

{validation_prompt}

{tool_prompt}

{code_improvement_prompt}

Please provide the complete Python script that accomplishes these tasks, ensuring it's ready to run given the appropriate data inputs.

### Task Description
{task_description}

### Data Structure
{data_prompt}

### User Instruction
{user_input}

### Previous Errors
These errors were encountered across different implementation approaches and may not be directly related to your current implementation. Use them as reference material to identify potential pitfalls and avoid similar mistakes in your implementation.
{all_error_analyses}

### Tutorials for Reference
{tutorial_prompt}

Please format your response with the code in a ```python``` code block to make it easily extractable.
"""

# ─── CoderAgent Prompts (Bash) ───

BASH_CODER_PROMPT = """\
Generate a minimal bash script that will:
{environment_prompt}
Execute the Python script: {python_file_path}

### Python code in the script:
{python_code}

### Previous Error
{all_error_analyses}

### Previous failed bash script:
{previous_bash_script}

Notes:
- Generate a minimal, executable bash script
- Focus on essential commands only
- Handle environment and package only if asked or there were errors

Please format your response with the code in a ```bash``` code block to make it easily extractable.
"""

# ─── ExecuterAgent Prompts ───

EXECUTER_PROMPT = """\
You are an expert code evaluator. Analyze the execution results of the following Python code and determine if the execution was successful or if issues need to be fixed.

### Task Descriptions
{task_description}

### Data Structure
{data_prompt}

### Python Code
{python_code}

## Execution Results
### Standard Output (stdout)

{stdout}

### Standard Error (stderr)

{stderr}

Evaluate the execution results and decide on one of the following actions:
1. SUCCESS - If the execution was completely successful and met all requirements.
2. FIX - If there were errors, issues, or performance problems that need to be addressed.

Provide your decision in the following format:
DECISION: [SUCCESS or FIX]
ERROR_SUMMARY: [Brief summary of errors if any, or "None" if no errors]
VALIDATION_SCORE: [If there is a validation score for the solution, provide it as a number, otherwise "None"]

The error summary should be brief but informative enough for another agent to understand what needs to be fixed.
Even if the code executed without throwing errors, it might still have issues with logic or not meet all requirements.

For validation scores:
- If there is a validation score present in the execution results, extract it (e.g. the last validation score reported in the training process).
- Convert the score to ensure higher values indicate better performance (multiply "lower is better" metrics like RMSE, MAE, or loss by -1)
- Return the converted score that follows the "higher is better" convention
"""

# ─── Environment Prompt Helper ───

def build_environment_prompt(
    docker_iter_folder: str,
    current_tool: str,
    common_req_file: str = "",
    tool_req_file: str = "",
    configure_env: bool = False,
) -> str:
    """Build the environment setup section for the bash coder prompt."""
    env_prompt = """Install required packages using uv in a virtual environment (we are running inside a sandbox container as user 'gem'):
  # Always ensure virtual environment is created and activated
  if [ ! -d "/home/gem/workspace/.venv" ]; then
      uv venv /home/gem/workspace/.venv
  fi
  source /home/gem/workspace/.venv/bin/activate
"""

    if common_req_file and tool_req_file:
        env_prompt += (
            f"  uv pip install --prerelease=allow -r \"{tool_req_file}\" -r \"{common_req_file}\"\n"
        )
    elif common_req_file:
        env_prompt += (
            f"  uv pip install --prerelease=allow -r \"{common_req_file}\"\n"
        )
    elif tool_req_file:
        env_prompt += (
            f"  uv pip install --prerelease=allow -r \"{tool_req_file}\"\n"
        )
    else:
        env_prompt += "  uv pip install --prerelease=allow <list the packages needed>\n"

    if configure_env:
        env_prompt += "  # Feel free to install any other python packages needed via `uv pip install`.\n"

    env_prompt += (
        "\nRun the Python script:\n"
        "  python \"{python_file_path}\"\n"
    )

    return env_prompt


def build_validation_prompt(continuous_improvement: bool) -> str:
    """Generate the validation section of the prompt."""
    if continuous_improvement:
        return """6. Validation (only when there is labeled training data):
   - If there is training and but no validation data is given, hold out a validation dataset (10 percent of the data) at the start, train only on the remaining data.
   - At the end compute and print the final evaluation metric score on the validation set.
   - Use a try-except block for the validation step - if validation fails, it's acceptable to continue.
"""
    return ""

# ─── Code Extraction ───

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

# ─── Requirements Resolver ───

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
