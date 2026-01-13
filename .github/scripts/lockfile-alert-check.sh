#!/usr/bin/env bash

# Determine whether the base branch has pixi.lock changes that the PR hasn't
# incorporated yet. Uses merge-base comparison: if the base branch's lockfile
# differs from the merge-base, the PR needs to rebase/merge.
#
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

# Fetch refs needed for merge-base calculation.
# - Base ref might be missing if PR targets a non-default branch
# - PR ref (refs/pull/N/head) is never auto-fetched by actions/checkout
git fetch --no-tags origin "${BASE_REF}"
git fetch --no-tags origin "pull/${PR_NUMBER}/head:pr-${PR_NUMBER}"

# Find the common ancestor between PR and base
MERGE_BASE=$(git merge-base "pr-${PR_NUMBER}" "${REMOTE_BASE}")

# Check if base branch has lockfile changes since the merge-base.
# If so, the PR needs to incorporate those changes.
# git diff --quiet exits 0 if no diff, 1 if diff exists, >1 on error
set +e
git diff --quiet "${MERGE_BASE}" "${REMOTE_BASE}" -- pixi.lock
diff_status=$?
set -e

if [[ ${diff_status} -gt 1 ]]; then
  echo "Error: git diff failed with status ${diff_status}" >&2
  exit 1
fi

if [[ ${diff_status} -eq 0 ]]; then
  result=false
  lockfile_commit=""
else
  result=true
  # Get the most recent commit on base branch that modified pixi.lock
  lockfile_commit=$(git log -1 --format='%H' "${MERGE_BASE}..${REMOTE_BASE}" -- pixi.lock)
fi

# Output results (always echo for debugging, write to GITHUB_OUTPUT if available)
echo "needs_alert=${result}"
echo "merge_base=${MERGE_BASE}"
echo "lockfile_commit=${lockfile_commit}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "needs_alert=${result}" >> "${GITHUB_OUTPUT}"
  echo "merge_base=${MERGE_BASE}" >> "${GITHUB_OUTPUT}"
  echo "lockfile_commit=${lockfile_commit}" >> "${GITHUB_OUTPUT}"
fi
