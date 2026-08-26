#!/usr/bin/env bash
# The loop lives in the verified-sandbox package now, and its knobs in
# [tool.sandbox] in pyproject.toml. This shim is here for muscle memory only.
exec sandbox "$@"
