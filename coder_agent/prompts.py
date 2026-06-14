"""
Prompt templates for the Coder Agent.
"""

# ─── CoderAgent (Python) ──────────────────────────────────────────────────

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


# ─── CoderAgent (Bash) ────────────────────────────────────────────────────

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


# ─── ExecuterAgent ────────────────────────────────────────────────────────

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


# ─── Environment prompt helper ───────────────────────────────────────────

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
    """Generate the validation section of the prompt, matching autogluon-assistant."""
    if continuous_improvement:
        return """6. Validation (only when there is labeled training data):
   - If there is training and but no validation data is given, hold out a validation dataset (10 percent of the data) at the start, train only on the remaining data.
   - At the end compute and print the final evaluation metric score on the validation set.
   - Use a try-except block for the validation step - if validation fails, it's acceptable to continue.
"""
    return ""
