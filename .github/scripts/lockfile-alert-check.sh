#!/usr/bin/env bash

# Determine whether pixi.lock differs between a PR head and its base branch.
# Arguments:
#   1: PR number
#   2: Base branch ref (e.g., main)

set -euo pipefail

PR_NUMBER="${1:-}"
BASE_REF="${2:-}"

if [[ -z "${PR_NUMBER}" || -z "${BASE_REF}" ]]; then
  echo "Usage: $0 <pr-number> <base-ref>" >&2
  exit 2
fi

PR_HEAD="refs/pull/${PR_NUMBER}/head"
REMOTE_BASE="origin/${BASE_REF}"

git fetch --no-tags --depth=1 origin "${BASE_REF}"
git fetch --no-tags --depth=1 origin "${PR_HEAD}"

# Compare pixi.lock between PR head and base without checking out
if git diff --quiet "${PR_HEAD}" "${REMOTE_BASE}" -- pixi.lock; then
  result=false
else
  result=true
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "needs_alert=${result}" >> "${GITHUB_OUTPUT}"
else
  echo "needs_alert=${result}"
fi

