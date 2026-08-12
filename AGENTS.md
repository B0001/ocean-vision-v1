# AGENTS.md Instructions

1. **Session Warm-up**: On startup, run `bd ready --json` to get the highest-priority, unblocked task.
2. **Execution**:
   - Focus *strictly* on implementing that single task.
   - Run tests/linting to verify your work.
3. **Session Tear-down**:
   - Mark the task complete: `bd close <task-id>` — only if the task's evidence exists and the suite passes. Otherwise leave it open with `--notes`.
   - Leave the tree ready to commit and report the commands. Do not commit or push; CLAUDE.md's conservative profile governs.
   - Do NOT pick up the next task in the same session. Exit gracefully.