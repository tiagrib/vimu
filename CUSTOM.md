# Project Custom Instructions

<!--
  Add your project-wide custom instructions here.
  This file is yours — `metak install --force` will never overwrite it.

  Use this for standing rules that apply to ALL agents (orchestrator + workers):
  - Tech stack preferences (e.g., "Use TypeScript strict mode everywhere")
  - Deployment targets and constraints (e.g., "All services deploy to AWS ECS")
  - Team conventions not captured in coding-standards.md
  - Domain context that agents need to keep in mind
-->

For python use pyenv-venv-win to manage virtual environments. Always create a new virtual environment for each project and activate it before installing dependencies. This helps to avoid conflicts between different projects and ensures that each project has its own isolated environment.

The command to list available python versions in pyenv-venv is:
```pyenv-venv list python
```

The command to list existing virtual environments in pyenv-venv is:
```pyenv-venv list envs
```

The command to install a new python version in pyenv is:
```pyenv install <version>
```

The command to create a new virtual environment in pyenv-venv is:
```pyenv-venv install <env-name> <python-version>
```

The command to activate a virtual environment in pyenv-venv is:
```pyenv-venv activate <env-name>
```
