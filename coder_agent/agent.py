"""
Single build agent compiling the Coder Agent StateGraph with inline nodes.
"""

import os
import logging
import time
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from .state import CoderAgentState
from .utils import extract_code, get_requirements_contents
from .prompts import (
    PYTHON_CODER_PROMPT,
    BASH_CODER_PROMPT,
    EXECUTER_PROMPT,
    build_environment_prompt,
    build_validation_prompt,
)

logger = logging.getLogger(__name__)

from common.metrics_context import MetricsContext
from common.metrics_emitter import node_metrics

metric_logger = logging.getLogger("agent_metrics")
ctx = MetricsContext(agent_id="coder_agent")

def build_coder_agent_graph(ctx=None, metric_logger=None):
    """
    Build and compile the Coder Agent StateGraph.
    
    Contains all graph nodes inline to encapsulate execution scope,
    matching the single build agent design pattern.
    """
    active_ctx = ctx or globals().get("ctx")
    active_logger = metric_logger or globals().get("metric_logger")

    async def _sandbox_tool_call(sandbox, tool_name: str, node_name: str, **kwargs):
        """
        Instrumented wrapper around sandbox operations.
        Emits tool_call events matching SessionMetricsCallback format.
        """
        run_id = str(uuid.uuid4())
        parent_run_id = str(uuid.uuid4())
        tool_input = json.dumps(kwargs, default=str)
        t0 = time.time()

        status = "success"
        output = ""
        error_str = None
        try:
            if tool_name == "sandbox_write_file":
                result = await sandbox.write_file(kwargs["path"], kwargs["content"])
                output = json.dumps({"path": kwargs["path"], "bytes_written": len(kwargs.get("content", ""))}, default=str)
            elif tool_name == "sandbox_exec_shell":
                success, stdout, stderr = await sandbox.exec_shell(
                    command=kwargs["command"],
                    cwd=kwargs.get("cwd", "/home/gem/workspace"),
                )
                result = (success, stdout, stderr)
                output = json.dumps({
                    "success": success,
                    "stdout_len": len(stdout),
                    "stderr_len": len(stderr),
                }, default=str)
            else:
                raise ValueError(f"Unknown sandbox tool: {tool_name}")
        except Exception as e:
            status = "error"
            error_str = str(e)
            raise
        finally:
            latency = (time.time() - t0) * 1000
            event = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
                "event_type": "tool_call",
                **(active_ctx.snapshot() if active_ctx else {}),
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "node_name": node_name,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "latency_ms": round(latency, 2),
                "status": status,
            }
            if status == "success":
                event["tool_output"] = output
            else:
                event["error"] = error_str
            if active_logger:
                active_logger.info(json.dumps(event))

        return result

    def _init_llm(llm_config: dict):
        """Helper to initialize ChatOpenAI or ChatOpenRouter directly from config."""
        model = llm_config.get("model", "deepseek/deepseek-v4-flash")
        temperature = llm_config.get("temperature", 0.1)
        max_tokens = llm_config.get("max_tokens", 32768)

        is_openai = model.lower().startswith("gpt") or model.lower().startswith("o1-") or model.lower().startswith("o3-")

        if is_openai:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Export it before running: export OPENAI_API_KEY=sk-..."
                )

            is_reasoning_model = any(x in model.lower() for x in ["o1-", "o3-", "gpt-5"])

            if is_reasoning_model:
                logger.info("Detected reasoning model. Forcing temp=1 and using max_completion_tokens.")
                return ChatOpenAI(
                    model=model,
                    temperature=1,
                    max_completion_tokens=max_tokens,
                    api_key=api_key,
                )
            
            logger.info(f"Initialized OpenAI LLM: model={model}, temp={temperature}")
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
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

    @node_metrics(active_ctx, active_logger, "generate_python_code")
    async def generate_python_code(state: CoderAgentState) -> dict:
        """LLM node to generate a python ML training script."""
        logger.info("─── [Coder Agent] generate_python_code ───")
        config = state.get("config", {})
        
        # LLM setup
        llm_config = config.get("llm", {}).copy()
        if "model" not in llm_config:
            llm_config["model"] = "gpt-4o"
        llm = _init_llm(llm_config)
        
        mcts_config = config.get("mcts", {})
        continuous_improvement = mcts_config.get("continuous_improvement", True)
        sandbox = state.get("sandbox_client")
        
        iteration = state.get("iteration", 0)
        node_id = state.get("node_id")

        # Build code improvement / debug context
        stage = state.get("stage", "")
        previous_code = state.get("previous_python_code", "")
        code_improvement_prompt = ""
        
        if stage == "debug" and previous_code:
            code_improvement_prompt = f"""\
### Previous Code to Debug
```python
{previous_code}
```
Please fix the errors in the code above. Make minimal changes necessary to fix the issues.
"""
            logger.info("  Mode: DEBUGGING previous code")
        elif stage == "evolve" and previous_code:
            code_improvement_prompt = f"""\
### Previous Code to Improve
```python
{previous_code}
```
Please prioritize model architecture improvements and training optimization to enhance performance.
"""
            logger.info("  Mode: IMPROVING previous code")

        # Validation prompt
        validation_prompt = build_validation_prompt(continuous_improvement)

        # Format all previous error analyses
        all_error_analyses = "\n\n".join(state.get("all_error_analyses", []))

        # Define paths inside the sandbox container
        if node_id is not None:
            iter_folder = f"/home/gem/workspace/node_{node_id}"
        else:
            iter_folder = f"/home/gem/workspace/iteration_{iteration}"
        per_iter_output = f"{iter_folder}/output"

        # Translate paths to sandbox path
        def _docker_translate(text: str) -> str:
            if not text:
                return ""
            text = text.replace("/home/gem/workspace", "PLACEHOLDER_HOME_GEM_WORKSPACE")
            text = text.replace("/workspace/data", "/home/gem/workspace/data")
            text = text.replace("/workspace/output", "/home/gem/workspace")
            text = text.replace("/workspace", "/home/gem/workspace")
            text = text.replace("PLACEHOLDER_HOME_GEM_WORKSPACE", "/home/gem/workspace")
            return text

        data_prompt = _docker_translate(state.get("data_prompt", ""))
        task_description = _docker_translate(state.get("task_description", ""))
        user_input = _docker_translate(state.get("user_input", ""))
        tutorial_prompt = _docker_translate(state.get("tutorial_prompt", ""))
        all_error_analyses = _docker_translate(all_error_analyses)

        prompt = PYTHON_CODER_PROMPT.format(
            current_tool=state.get("current_tool", ""),
            output_folder=per_iter_output,
            tool_prompt=state.get("tool_prompt", ""),
            code_improvement_prompt=code_improvement_prompt,
            validation_prompt=validation_prompt,
            task_description=task_description,
            data_prompt=data_prompt,
            user_input=user_input,
            all_error_analyses=all_error_analyses or "None",
            tutorial_prompt=tutorial_prompt or "None",
        )

        response = await llm.ainvoke(prompt)
        response_text = response.content
        if active_ctx and active_logger:
            from common.metrics_emitter import emit_event
            emit_event(active_logger, {
                **active_ctx.snapshot(),
                "event_type": "debug",
                "node_name": "generate_python_code",
                "iteration": state.get("iteration", 0),
                "stage": state.get("stage", "root"),
                "prompt": prompt,
                "response": response_text,
                "prompt_len": len(prompt),
                "response_len": len(response_text),
            })
        python_code = extract_code(response_text, language="python")

        # Save to file inside the sandbox
        python_file_path = f"{iter_folder}/generated_code.py"
        if sandbox:
            await _sandbox_tool_call(
                sandbox, "sandbox_write_file", "generate_python_code",
                path=python_file_path, content=python_code,
            )
            logger.info(f"  Python code saved inside sandbox: {python_file_path} ({len(python_code)} chars)")
        else:
            logger.warning("No sandbox client found in state while saving Python code!")

        return {"python_code": python_code, "python_file_path": python_file_path}

    @node_metrics(active_ctx, active_logger, "generate_bash_script")
    async def generate_bash_script(state: CoderAgentState) -> dict:
        """LLM node to generate a bash execution script."""
        logger.info("─── [Coder Agent] generate_bash_script ───")
        config = state.get("config", {})
        
        # LLM setup
        llm_config = config.get("llm", {}).copy()
        if "model" not in llm_config:
            llm_config["model"] = "gpt-4o"
        llm = _init_llm(llm_config)
        
        sandbox = state.get("sandbox_client")
        current_tool = state.get("current_tool", "")
        iteration = state.get("iteration", 0)
        node_id = state.get("node_id")

        if node_id is not None:
            iter_folder = f"/home/gem/workspace/node_{node_id}"
        else:
            iter_folder = f"/home/gem/workspace/iteration_{iteration}"

        python_file_path = state.get("python_file_path", "")
        if not python_file_path:
            python_file_path = f"{iter_folder}/generated_code.py"

        # Copy requirements files from registry on host into the sandbox via MCP write
        registry_path = config.get("tool_registry_path")
        common_req_content, tool_req_content = get_requirements_contents(registry_path, current_tool)
        
        docker_common_req = ""
        docker_tool_req = ""
        if sandbox:
            if common_req_content:
                try:
                    await _sandbox_tool_call(
                        sandbox, "sandbox_write_file", "generate_bash_script",
                        path=f"{iter_folder}/requirements_common.txt", content=common_req_content,
                    )
                    docker_common_req = f"{iter_folder}/requirements_common.txt"
                except Exception as e:
                    logger.warning(f"Failed to write requirements_common.txt to sandbox: {e}")
            if tool_req_content:
                try:
                    await _sandbox_tool_call(
                        sandbox, "sandbox_write_file", "generate_bash_script",
                        path=f"{iter_folder}/requirements_tool.txt", content=tool_req_content,
                    )
                    docker_tool_req = f"{iter_folder}/requirements_tool.txt"
                except Exception as e:
                    logger.warning(f"Failed to write requirements_tool.txt to sandbox: {e}")

        # Determine if env configuration is needed (open-ended install)
        configure_env = current_tool.lower() in [
            "machine_learning", "machine learning", "huggingface", "fairseq"
        ]

        environment_prompt = build_environment_prompt(
            docker_iter_folder=iter_folder,
            current_tool=current_tool,
            common_req_file=docker_common_req,
            tool_req_file=docker_tool_req,
            configure_env=configure_env,
        )

        def _docker_translate(text: str) -> str:
            if not text:
                return ""
            text = text.replace("/home/gem/workspace", "PLACEHOLDER_HOME_GEM_WORKSPACE")
            text = text.replace("/workspace/data", "/home/gem/workspace/data")
            text = text.replace("/workspace/output", "/home/gem/workspace")
            text = text.replace("/workspace", "/home/gem/workspace")
            text = text.replace("PLACEHOLDER_HOME_GEM_WORKSPACE", "/home/gem/workspace")
            return text

        all_error_analyses = "\n\n".join(state.get("all_error_analyses", []))
        all_error_analyses = _docker_translate(all_error_analyses)

        prompt = BASH_CODER_PROMPT.format(
            environment_prompt=environment_prompt,
            python_file_path=python_file_path,
            python_code=state.get("python_code", ""),
            all_error_analyses=all_error_analyses or "None",
            previous_bash_script=state.get("previous_bash_script", "") or "None",
        )

        response = await llm.ainvoke(prompt)
        response_text = response.content
        if active_ctx and active_logger:
            from common.metrics_emitter import emit_event
            emit_event(active_logger, {
                **active_ctx.snapshot(),
                "event_type": "debug",
                "node_name": "generate_bash_script",
                "iteration": state.get("iteration", 0),
                "prompt": prompt,
                "response": response_text,
                "prompt_len": len(prompt),
                "response_len": len(response_text),
            })
        bash_script = extract_code(response_text, language="bash")

        # Save to file inside sandbox
        bash_file_path = f"{iter_folder}/execution_script.sh"
        if sandbox:
            await _sandbox_tool_call(
                sandbox, "sandbox_write_file", "generate_bash_script",
                path=bash_file_path, content=bash_script,
            )
            logger.info(f"  Bash script saved inside sandbox: {bash_file_path} ({len(bash_script)} chars)")
        else:
            logger.warning("No sandbox client found in state while saving bash script!")

        return {"bash_script": bash_script}

    @node_metrics(active_ctx, active_logger, "execute_and_evaluate")
    async def execute_and_evaluate(state: CoderAgentState) -> dict:
        """LLM node to run the script inside sandbox and evaluate results."""
        logger.info("─── [Coder Agent] execute_and_evaluate ───")
        config = state.get("config", {})
        
        # LLM setup
        llm_config = config.get("llm", {}).copy()
        if "model" not in llm_config:
            llm_config["model"] = "gpt-4o"
        llm = _init_llm(llm_config)
        
        sandbox = state.get("sandbox_client")
        iteration = state.get("iteration", 0)
        node_id = state.get("node_id")

        if node_id is not None:
            iter_folder = f"/home/gem/workspace/node_{node_id}"
        else:
            iter_folder = f"/home/gem/workspace/iteration_{iteration}"

        # Execute inside Sandbox
        start_exec = time.time()
        if sandbox:
            success, stdout, stderr = await _sandbox_tool_call(
                sandbox, "sandbox_exec_shell", "execute_and_evaluate",
                command="bash execution_script.sh",
                cwd=iter_folder,
            )
        else:
            success, stdout, stderr = False, "", "No sandbox client found in state"

        exec_time = time.time() - start_exec
        logger.info(f"  Execution {'SUCCEEDED' if success else 'FAILED'} (took {exec_time:.2f}s)")

        # Save raw output inside the sandbox
        if sandbox:
            try:
                await _sandbox_tool_call(
                    sandbox, "sandbox_write_file", "execute_and_evaluate",
                    path=f"{iter_folder}/stdout.txt", content=stdout,
                )
                await _sandbox_tool_call(
                    sandbox, "sandbox_write_file", "execute_and_evaluate",
                    path=f"{iter_folder}/stderr.txt", content=stderr,
                )
            except Exception as e:
                logger.warning(f"Failed to write stdout/stderr logs in sandbox: {e}")

        # Truncate for LLM
        def truncate_start(text, max_len=8192):
            if len(text) > max_len:
                return f"[...TRUNCATED ({len(text) - max_len} chars)...]\n" + text[-max_len:]
            return text

        def _docker_translate(text: str) -> str:
            if not text:
                return ""
            text = text.replace("/home/gem/workspace", "PLACEHOLDER_HOME_GEM_WORKSPACE")
            text = text.replace("/workspace/data", "/home/gem/workspace/data")
            text = text.replace("/workspace/output", "/home/gem/workspace")
            text = text.replace("/workspace", "/home/gem/workspace")
            text = text.replace("PLACEHOLDER_HOME_GEM_WORKSPACE", "/home/gem/workspace")
            return text

        prompt = EXECUTER_PROMPT.format(
            task_description=_docker_translate(state.get("task_description", "")),
            data_prompt=_docker_translate(state.get("data_prompt", "")),
            python_code=state.get("python_code", ""),
            stdout=truncate_start(stdout) or "No standard output",
            stderr=truncate_start(stderr) or "No standard error",
        )

        response = await llm.ainvoke(prompt)
        content = response.content
        if active_ctx and active_logger:
            from common.metrics_emitter import emit_event
            emit_event(active_logger, {
                **active_ctx.snapshot(),
                "event_type": "debug",
                "node_name": "execute_and_evaluate",
                "iteration": state.get("iteration", 0),
                "prompt": prompt,
                "response": content,
                "prompt_len": len(prompt),
                "response_len": len(content),
            })

        # Parse decision
        decision = "FIX"
        if "DECISION:" in content:
            for line in content.split("\n"):
                if "DECISION:" in line:
                    if "SUCCESS" in line.upper():
                        decision = "SUCCESS"
                    break

        # Parse error summary
        error_summary = None
        if "ERROR_SUMMARY:" in content:
            es = content.split("ERROR_SUMMARY:")[1].strip().split("\n")[0].strip()
            if es.lower() != "none" and es:
                error_summary = es

        # Parse validation score
        validation_score = None
        if "VALIDATION_SCORE:" in content:
            vs_text = content.split("VALIDATION_SCORE:")[1].strip().split("\n")[0].strip()
            if vs_text.lower() != "none" and vs_text:
                try:
                    validation_score = float(vs_text)
                except ValueError:
                    pass
        if decision != "SUCCESS":
            validation_score = None

        error_message = ""
        if stderr:
            error_message = f"stderr: {stderr}\n\n"
        if error_summary:
            error_message += f"Error summary: {error_summary}"

        logger.info(f"  Decision: {decision}")
        logger.info(f"  Validation score: {validation_score}")

        return {
            "stdout": stdout,
            "stderr": stderr,
            "decision": decision,
            "error_summary": error_summary,
            "validation_score": validation_score,
            "error_message": error_message,
        }

    # Graph Setup
    graph = StateGraph(CoderAgentState)
    graph.add_node("generate_python_code", generate_python_code)
    graph.add_node("generate_bash_script", generate_bash_script)
    graph.add_node("execute_and_evaluate", execute_and_evaluate)

    graph.add_edge(START, "generate_python_code")
    graph.add_edge("generate_python_code", "generate_bash_script")
    graph.add_edge("generate_bash_script", "execute_and_evaluate")
    graph.add_edge("execute_and_evaluate", END)

    return graph.compile()
