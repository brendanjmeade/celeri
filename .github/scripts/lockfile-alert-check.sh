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

REMOTE_BASE="origin/${BASE_REF}"

git fetch --no-tags --depth=1 origin "${BASE_REF}"
git fetch --no-tags --depth=1 origin "pull/${PR_NUMBER}/head:pr-${PR_NUMBER}"

# Compare pixi.lock between PR head and base without checking out
# git diff --quiet exits 0 if no diff, 1 if diff exists, >1 on error
set +e
git diff --quiet "pr-${PR_NUMBER}" "${REMOTE_BASE}" -- pixi.lock
diff_status=$?
set -e

if [[ ${diff_status} -gt 1 ]]; then
  echo "Error: git diff failed with status ${diff_status}" >&2
  exit 1
fi

if [[ ${diff_status} -eq 0 ]]; then
  result=false
else
  result=true
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "needs_alert=${result}" >> "${GITHUB_OUTPUT}"
else
  echo "needs_alert=${result}"
fi

