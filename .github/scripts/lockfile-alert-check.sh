#!/usr/bin/env bash

# Determine whether the base branch has pixi.lock changes that the PR hasn't
# incorporated yet, and whether the PR itself modifies pixi.lock.
#
# Outputs an alert_level:
#   - "none": No action needed (base has no lockfile changes since merge-base)
#   - "warning": Base has lockfile changes, but PR doesn't touch pixi.lock
#                (informational - merge before making lockfile changes)
#   - "error": Both base and PR modify pixi.lock (conflict must be resolved now)
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
# git diff --quiet exits 0 if no diff, 1 if diff exists, >1 on error
set +e
git diff --quiet "${MERGE_BASE}" "${REMOTE_BASE}" -- pixi.lock
base_diff_status=$?
set -e

if [[ ${base_diff_status} -gt 1 ]]; then
  echo "Error: git diff (base) failed with status ${base_diff_status}" >&2
  exit 1
fi

# Check if PR has lockfile changes since the merge-base.
set +e
git diff --quiet "${MERGE_BASE}" "pr-${PR_NUMBER}" -- pixi.lock
pr_diff_status=$?
set -e

if [[ ${pr_diff_status} -gt 1 ]]; then
  echo "Error: git diff (PR) failed with status ${pr_diff_status}" >&2
  exit 1
fi

base_modified_lockfile=$([[ ${base_diff_status} -eq 1 ]] && echo "true" || echo "false")
pr_modified_lockfile=$([[ ${pr_diff_status} -eq 1 ]] && echo "true" || echo "false")

# If both modified, check if the final contents are identical (no actual conflict)
lockfiles_identical="false"
if [[ "${base_modified_lockfile}" == "true" && "${pr_modified_lockfile}" == "true" ]]; then
  set +e
  git diff --quiet "pr-${PR_NUMBER}" "${REMOTE_BASE}" -- pixi.lock
  identical_status=$?
  set -e
  if [[ ${identical_status} -eq 0 ]]; then
    lockfiles_identical="true"
  fi
fi

# Determine alert level based on checks
if [[ "${base_modified_lockfile}" == "false" ]]; then
  # Base hasn't changed lockfile since merge-base, nothing to do
  alert_level="none"
  lockfile_commit=""
elif [[ "${pr_modified_lockfile}" == "true" && "${lockfiles_identical}" == "false" ]]; then
  # Both modified and contents differ - conflict will occur
  alert_level="error"
  lockfile_commit=$(git log -1 --format='%H' "${MERGE_BASE}..${REMOTE_BASE}" -- pixi.lock)
else
  # Either:
  #   1. Only base modified the lockfile, or
  #   2. Both modified but contents are identical (edge case: e.g., both branches
  #      added the same dependency independently). No conflict exists *now*, but
  #      any future lockfile regeneration on the PR would create one.
  # In both cases, warn so the author merges before making further lockfile changes.
  alert_level="warning"
  lockfile_commit=$(git log -1 --format='%H' "${MERGE_BASE}..${REMOTE_BASE}" -- pixi.lock)
fi

# Output results (always echo for debugging, write to GITHUB_OUTPUT if available)
echo "alert_level=${alert_level}"
echo "base_modified_lockfile=${base_modified_lockfile}"
echo "pr_modified_lockfile=${pr_modified_lockfile}"
echo "lockfiles_identical=${lockfiles_identical}"
echo "merge_base=${MERGE_BASE}"
echo "lockfile_commit=${lockfile_commit}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "alert_level=${alert_level}" >> "${GITHUB_OUTPUT}"
  echo "base_modified_lockfile=${base_modified_lockfile}" >> "${GITHUB_OUTPUT}"
  echo "pr_modified_lockfile=${pr_modified_lockfile}" >> "${GITHUB_OUTPUT}"
  echo "merge_base=${MERGE_BASE}" >> "${GITHUB_OUTPUT}"
  echo "lockfile_commit=${lockfile_commit}" >> "${GITHUB_OUTPUT}"
fi
