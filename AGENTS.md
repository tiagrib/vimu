# MetaKitchen Agent Guide

This is a multi-repository workspace. Each sub-repo has its own `.git` and may have its own agent instruction files.

## Structure

```
meta-repo/
├── AGENTS.md                    ← you are here — start by reading this file
├── metak-shared/                ← shared context (architecture, API contracts, glossary)
├── metak-orchestrator/          ← coordination workspace (TASKS.md, STATUS.md, EPICS.md)
├── repo-*/                      ← application sub-repos
└── meta.code-workspace          ← VS Code multi-root workspace file
```

## Agent Roles

### Orchestrator

The orchestrator agent (running from `metak-orchestrator/`) coordinates all cross-repo work. It:

- **Creates and maintains `metak-shared/` docs** (overview, architecture, API contracts, glossary) based on user input, requesting user review for each.
- **Creates new workspace subfolders** via `metak add` (or manually if the CLI is unavailable) when the project requires them.
- **Breaks work into epics and tasks**, delegates to worker agents, and monitors progress.

See `metak-orchestrator/AGENTS.md` for full orchestrator instructions.

### Worker Agents

Worker agents operate within a single sub-repo. They:

- Read their assignment from `metak-orchestrator/TASKS.md`.
- Update `metak-orchestrator/STATUS.md` when done or blocked.
- **Treat `metak-shared/` as read-only.** Never modify shared docs — propose changes via the orchestrator for user review.

## Agent Rules

1. **Read this file and any agent instructions in your working repo before starting work.**
2. **One agent, one subfolder, one repo.** Do not work across multiple workspace subfolders or repos in a single session. Use the orchestrator pattern for cross-repo work.
3. **API contracts live in `metak-shared/api-contracts/`.** Always reference these for schemas — never import directly from another repo's source.
4. **Consult `metak-shared/architecture.md`** for system boundaries and service interactions.

## Coding Standards

- Follow the coding standards defined in `metak-shared/coding-standards.md` for your repo's language. This includes linting rules, commit message conventions, and testing expectations.

## Custom Instructions

Read and follow `CUSTOM.md` at the project root for project-specific instructions that apply to all agents.
