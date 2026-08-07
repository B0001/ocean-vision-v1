#!/usr/bin/env bash
# iterate-beads.sh

while true; do
    # 1. Fetch the next ready task ID from Beads
    TASK_ID=$(bd ready --json | jq -r '.[0].id // empty')

    # 2. If no tasks are ready, exit loop
    if [ -z "$TASK_ID" ]; then
        echo "🎉 All Beads tasks completed or blocked!"
        break
    fi

    echo "🚀 Starting fresh Claude Code worker for task: $TASK_ID"

    # 3. Launch a fresh, headless Claude Code session targeting the task
    claude -p "Your task is to complete Beads issue $TASK_ID. 
               Run 'bd show $TASK_ID' to read the full specs and dependencies. 
               Implement the solution, run all unit tests to confirm it works, 
               close the task with 'bd close $TASK_ID', and then exit."

    # 4. Cleanup/Sync git and Beads state between sessions
    bd doctor --fix
    bd sync
    
    echo "✅ Worker finished $TASK_ID. Starting next cycle..."
    sleep 2
done