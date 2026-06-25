_QUERY_GENERATOR_PROMPT = """\
You are an expert at generating search queries to find relevant machine learning tutorials. Given the context below, generate a concise and effective search query that will help find the most relevant tutorials for this task.

### Task Description
{task_description}

### Data Structures
{data_prompt}

### User Instruction
{user_input}

### Previous Error Analysis
{all_previous_error_analyses}

### Selected Tool/Library
{selected_tool}


Based on the above context, generate a search query that will help find tutorials most relevant to this task. The query should:
1. Include key technical terms and concepts
2. Focus on the main task/problem to solve
3. Be concise but specific

IMPORTANT: Respond ONLY with the search query text. Do not include explanations, quotes, or any other formatting.
"""

_RERANKER_PROMPT = """\
Given the following context and list of tutorials with their summaries, select the {max_num_tutorials} most relevant tutorials for helping with this task. Consider how well each tutorial's title and summary match the task, data, user question, and any errors.

### Task Description
{task_description}

### Data Structures
{data_prompt}

### User Instruction
{user_input}

### Previous Error Analysis
{all_previous_error_analyses}

Available Tutorials:
{tutorials_info}

IMPORTANT: Respond ONLY with the numbers of the selected tutorials (up to {max_num_tutorials}) separated by commas. 
For example: "1,3,4" or "2,5" or just "1" if only one is relevant.
DO NOT include any other text, explanation, or formatting in your response.
"""
