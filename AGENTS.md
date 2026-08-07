# AGENTS.md Instructions

1. **Session Warm-up**: On startup, run `bd ready --json` to get the highest-priority, unblocked task.
2. **Execution**:
   - Focus *strictly* on implementing that single task.
   - Run tests/linting to verify your work.
3. **Session Tear-down**:
   - Mark the task complete: `bd close <task-id>`
   - Commit changes to git.
   - Do NOT pick up the next task in the same session. Exit gracefully.