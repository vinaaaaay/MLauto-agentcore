# MLauto-agentcore

## Overview

MLauto-agentcore is a distributed, multi-agent orchestration framework designed for automated software engineering and complex reasoning tasks. Built on top of `bedrock-agentcore` and `langgraph`, the framework leverages a microservices architecture to decouple specialized agent capabilities, enabling scalable and modular execution of machine learning and development workflows.

## System Architecture

The repository is structured around a central orchestrator and several specialized peripheral services, each deployable independently.

### Core Components

*   **MLorchestrator**: The central control plane responsible for coordinating agent interactions, managing state transitions via LangGraph, and interfacing with isolated sandbox environments. It routes tasks to appropriate specialized agents and aggregates results.
*   **coder_agent**: A specialized agent dedicated to software generation, code modification, and sandbox synchronization.
*   **perception_agent**: An agent focused on environment observation, test execution analysis, and interpreting sandbox states through a registered tools registry.
*   **semantic_agent**: Handles semantic reasoning, code understanding, and contextual search operations to support the coder and orchestrator.
*   **mcts_handler**: Implements Monte Carlo Tree Search (MCTS) capabilities to facilitate advanced planning, exploring multiple solution paths and optimizing decision-making processes.
*   **mcpserver**: A Model Context Protocol (MCP) server integration that standardizes tool exposure and usage across the agent ecosystem.

## Technical Stack

*   **Language**: Python 3.10+
*   **Core Frameworks**: 
    *   `bedrock-agentcore` (>=1.3.1)
    *   `langgraph` & `langchain` for agent state management and LLM orchestration
    *   `mcp` for standardized context and tool protocol
*   **Package Management**: `uv` (project defined in `pyproject.toml` and locked via `uv.lock`)

## Repository Structure

```text
MLauto-agentcore/
├── MLorchestrator/        # State machine, routing, and sandbox orchestration
├── coder_agent/           # Code generation and file manipulation agent
├── perception_agent/      # Environment state and test result analysis agent
├── semantic_agent/        # Codebase context and semantic reasoning agent
├── mcts_handler/          # Monte Carlo Tree Search planning service
├── mcpserver/             # Model Context Protocol integration
└── ...                    # Root-level metric analysis and plotting scripts
```

## Setup and Installation

This project utilizes `uv` for dependency management. To set up the local environment:

1.  Ensure Python 3.10 or higher is installed.
2.  Install `uv` if not already present in the system.
3.  Sync the environment dependencies:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip sync uv.lock
    ```

    Alternatively, for a standard pip installation using the `pyproject.toml`:
    ```bash
    pip install -e .
    ```

## Deployment

Each core component is designed as a standalone microservice. Subdirectories (`MLorchestrator`, `coder_agent`, etc.) contain localized deployment scripts (`deploy.sh`), `Dockerfile` definitions, and AWS Bedrock Agent Core configurations (`.bedrock_agentcore.yaml`). 

To deploy a specific component, navigate to its respective directory and utilize the provided deployment shell script.

## Metrics and Telemetry

The root directory contains several utility scripts for telemetry and run analysis:
*   `analyze_run_metrics.py` / `analyze_run_metrics_yaml.py`: Parse logs and evaluate performance characteristics of agent runs.
*   `calculate_compute_costs.py`: Computes LLM inference and API costs associated with the orchestration workflows.
*   `plot_coder_invocations.py`: Generates visualizations of agent activity, token usage, and resource utilization.
