"""
Prompt templates for the Perception Agent.

Each template is used by a specific LangGraph node:
  - PYTHON_READER_PROMPT              → scan_data (file reading via LLM)
  - DESCRIPTION_FILE_RETRIEVER_PROMPT → find_description_files
  - TASK_DESCRIPTOR_PROMPT            → generate_task_description
  - TOOL_SELECTOR_PROMPT              → select_tools
"""

# ─── DataPerceptionAgent: reads individual files via LLM ──────────────────

PYTHON_READER_PROMPT = """\
Generate Python code to read and analyze the file: "{file_path}"

File Size: {file_size_mb} MB

Your code should:
1. Import all modules used (e.g. import os).
2. Use appropriate libraries based on file type (pandas for tabular data, etc.)
3. For tabular files (csv, excel, parquet, etc.):
    - Display column names. If there are more than 20 columns, only display the first and last 10.
    - Show first 2-3 rows with truncated cell content (50 chars).
    - Do not show additional index column if it's not in the original table.
    - If failed to open the file, treat it as text file.
4. For text files:
    - Display first few lines (up to {max_chars} characters).
5. For compressed tabular or text files, show its decompressed content as described.
6. For binary or other files, provide only the most basic information.
7. Keep the total output under {max_chars} characters.

Return ONLY the Python code, no explanations. The code should be self-contained and executable.

Please format your response with the code in a ```python``` code block to make it easily extractable.
"""


# ─── DescriptionFileRetrieverAgent ────────────────────────────────────────

DESCRIPTION_FILE_RETRIEVER_PROMPT = """\
Given the data structure, please identify any files that appear to contain project descriptions, requirements, or task definitions.
Look for files like README, documentation files, or task description files.

### Data Structure
{data_prompt}

Format your response as follows, do not give explanations:
Description Files: [list ONLY the absolute path, one per line]
"""


# ─── TaskDescriptorAgent ─────────────────────────────────────────────────

TASK_DESCRIPTOR_PROMPT = """\
Based ONLY on the information explicitly stated in the provided data structure and description files, provide a condensed and precise description of the data science task. Include only details that are directly mentioned in the source materials. Do not add assumptions or infer unstated information.

Be very clear about the problem type (e.g. audio classification/image regression/seq-to-seq generation/etc.), input format, and prediction output format.

### User Instruction
{user_input}

### Data Structure:
(IMPORTANT: The metadata of example files in Data Structure may not be representative - do not make assumptions about data statistics based on examples.)
{data_prompt}

### Description File Contents:
{description_file_contents}
"""


# ─── ToolSelectorAgent ───────────────────────────────────────────────────

TOOL_SELECTOR_PROMPT = """\
You are a data science expert tasked with selecting and ranking the most appropriate ML libraries for a specific task.

### Task Description:
{task_description}

### Data Information:
{data_prompt}

### Available ML Libraries:
{tools_info}

IMPORTANT: Your response MUST follow this exact format:
---
EXPLANATION: <provide your detailed reasoning process for evaluating the libraries>

RANKED_LIBRARIES:
1. <first choice library name>
2. <second choice library name>
3. <third choice library name>
...
---

Requirements for your response:
1. First provide a detailed explanation of your reasoning process using the "EXPLANATION:" header
2. Then provide a ranking of libraries using the "RANKED_LIBRARIES:" header
3. The library names must be exactly as shown in the available libraries list
4. Provide a ranking of at least 3 libraries (if available)
5. In your explanation, analyze each library's strengths and weaknesses for this specific task
6. Consider the task requirements, data characteristics, and library features

Do not include any other formatting or additional sections in your response.
"""
