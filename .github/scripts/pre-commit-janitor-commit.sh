#!/usr/bin/env bash

# Create the first commit for pre-commit autofixes and prepare .git-blame-ignore-revs.
#
# This script:
#   1. Checks if there are any staged/unstaged changes
#   2. Creates a commit with all changes ("pre-commit autofix")
#   3. Appends the commit hash to .git-blame-ignore-revs (leaves it uncommitted
#      for peter-evans/create-pull-request to commit)
#
# Outputs:
#   - has_changes: "true" if changes were committed, "false" otherwise

set -euo pipefail

# Check for changes
if git diff --quiet && git diff --cached --quiet; then
  echo "No changes from pre-commit autofixes"
  echo "has_changes=false"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "has_changes=false" >> "${GITHUB_OUTPUT}"
  fi
  exit 0
fi

echo "Pre-commit made changes, creating commit..."

# Configure git identity
git config user.name "pre-commit-janitor.yaml"
git config user.email "nobody@example.com"

# Create commit with all autofixes
git add -A
git commit -m "pre-commit autofix"
AUTOFIX_COMMIT=$(git rev-parse HEAD)
echo "Created autofix commit: ${AUTOFIX_COMMIT}"

# Append to .git-blame-ignore-revs (create if it doesn't exist)
# Leave uncommitted for peter-evans/create-pull-request to handle
if [[ ! -f .git-blame-ignore-revs ]]; then
  cat > .git-blame-ignore-revs <<'EOF'
# This file lists revisions that should be ignored when blaming a file.
# It should include all substantial machine-generated reformatting commits.
EOF
else
  echo "" >> .git-blame-ignore-revs
fi
{
  echo "# pre-commit autofix"
  echo "${AUTOFIX_COMMIT}"
} >> .git-blame-ignore-revs
echo "Updated .git-blame-ignore-revs"

# Output results
echo "has_changes=true"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "has_changes=true" >> "${GITHUB_OUTPUT}"
fi
