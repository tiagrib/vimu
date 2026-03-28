# Coding Standards

- All code must pass linting and tests before committing.
- Never import directly from another repo's source code. Use shared contracts in `metak-shared/api-contracts/`.
- When in doubt about system boundaries, consult `metak-shared/architecture.md`.

## Commit Messages

Follow Conventional Commits: `type(scope): description`

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Code Review

- All changes go through PRs.
- At least one human approval required before merge.
- CI must pass.
