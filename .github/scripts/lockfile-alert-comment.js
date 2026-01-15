#!/usr/bin/env node

/**
 * Manage lockfile alert comments on a pull request.
 *
 * Environment variables:
 *   ALERT_LEVEL - "none", "warning", or "error"
 *     - none: No alert needed (base has no lockfile changes)
 *     - warning: Base has lockfile changes, and either PR doesn't modify
 *                pixi.lock, or both modified but resulting contents are identical
 *     - error: Both base and PR modify pixi.lock with differing contents
 *   MERGE_BASE - Merge base commit SHA
 *   LOCKFILE_COMMIT - Most recent commit that modified pixi.lock on base branch
 *   PR_NUMBER - Pull request number
 *   BASE_REF - Base branch reference (e.g., "main")
 *   AUTHOR - PR author's GitHub username
 *   GITHUB_TOKEN - GitHub token with issues:write permissions
 *   GITHUB_REPOSITORY - Repository in "owner/repo" format
 */

const https = require('https');

const MARKER = '<!-- lockfile-alert -->';

const alertLevel = (process.env.ALERT_LEVEL || 'none').toLowerCase();
const mergeBase = process.env.MERGE_BASE || '';
const lockfileCommit = process.env.LOCKFILE_COMMIT || '';
const prNumber = Number(process.env.PR_NUMBER);
const baseRef = process.env.BASE_REF;
const headRef = process.env.HEAD_REF;
const author = process.env.AUTHOR;
const token = process.env.GITHUB_TOKEN;
const [owner, repo] = (process.env.GITHUB_REPOSITORY || '').split('/');
const defaultBranch = process.env.DEFAULT_BRANCH || 'main';
const workflowUrl = `https://github.com/${owner}/${repo}/blob/${defaultBranch}/.github/workflows/lockfile-alert.yaml`;
const headRepo = process.env.HEAD_REPO;
const baseRepo = process.env.BASE_REPO;

if (!prNumber || !baseRef || !headRef || !author || !token || !owner || !repo) {
  console.error('Missing required environment variables');
  process.exit(1);
}

// Determine the correct remote name based on whether this is a fork
let remoteName = 'origin';
let remoteContext = '';

if (headRepo && baseRepo && headRepo !== baseRepo) {
  remoteName = 'upstream';
  remoteContext = `\n\nRemote \`upstream\` should point at ${baseRepo}; your fork is ${headRepo}.`;
} else if (baseRepo) {
  remoteContext = `\n\nRemote \`origin\` should point at ${baseRepo} because this branch lives in ${baseRepo}.`;
}

// Warning message: base has lockfile changes, but PR doesn't touch pixi.lock
// This is informational - no immediate action required
const warningBody = `${MARKER}
> [!NOTE]
> **Heads up, @${author}!** \`${baseRef}\` has updated \`pixi.lock\` since this branch diverged.

This PR doesn't modify \`pixi.lock\`, so no conflict exists yet. However, **if you need to make any changes to \`pixi.lock\`**, be sure to merge or rebase \`${baseRef}\` first to avoid conflicts:${remoteContext}

\`\`\`bash
git checkout ${headRef}
git fetch ${remoteName}
git merge ${remoteName}/${baseRef}
\`\`\`

Merge-base: ${mergeBase}, latest lockfile change on \`${baseRef}\`: ${lockfileCommit}.

This notice will be minimized automatically once \`pixi.lock\` on \`${baseRef}\` matches the merge-base.

---
_Generated automatically by [\`.github/workflows/lockfile-alert.yaml\`](${workflowUrl})._
`;

// Error message: both base and PR modify pixi.lock - conflict must be resolved
const errorBody = `${MARKER}
> [!WARNING]
> **Action required, @${author}!** Both this PR and \`${baseRef}\` have modified \`pixi.lock\`.

You must merge or rebase \`${baseRef}\` to resolve the conflict before this PR can be merged.${remoteContext}

\`\`\`bash
git checkout ${headRef}
git fetch ${remoteName}
git merge ${remoteName}/${baseRef}
\`\`\`

To resolve the \`pixi.lock\` conflict, accept the incoming changes from \`${baseRef}\` and regenerate the lockfile:

\`\`\`bash
git checkout --theirs pixi.lock
pixi lock
git add pixi.lock
# resolve any other conflicts
git merge --continue
\`\`\`

Merge-base: ${mergeBase}, latest lockfile change on \`${baseRef}\`: ${lockfileCommit}.

This alert will be minimized automatically once \`pixi.lock\` conflicts are resolved.

---
_Generated automatically by [\`.github/workflows/lockfile-alert.yaml\`](${workflowUrl})._
`;

// Select the appropriate message body based on alert level
const alertBody = alertLevel === 'error' ? errorBody : warningBody;

/**
 * Make a GitHub API request
 */
function githubRequest(method, path, body = null) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.github.com',
      path,
      method,
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'lockfile-alert-script',
        'X-GitHub-Api-Version': '2022-11-28',
      },
    };

    if (body) {
      options.headers['Content-Type'] = 'application/json';
    }

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(data ? JSON.parse(data) : null);
        } else {
          reject(new Error(`HTTP ${res.statusCode}: ${data}`));
        }
      });
    });

    req.on('error', reject);
    if (body) {
      req.write(JSON.stringify(body));
    }
    req.end();
  });
}

/**
 * Make a GitHub GraphQL request
 */
function githubGraphQL(query, variables = {}) {
  return githubRequest('POST', '/graphql', { query, variables });
}

/**
 * Find all comments on the PR and return the one with our marker
 */
async function findAlertComment() {
  let page = 1;
  const perPage = 100;

  while (true) {
    const comments = await githubRequest(
      'GET',
      `/repos/${owner}/${repo}/issues/${prNumber}/comments?per_page=${perPage}&page=${page}`
    );

    if (!comments || comments.length === 0) {
      return null;
    }

    const found = comments.find(c => c.body && c.body.includes(MARKER));
    if (found) {
      return found;
    }

    if (comments.length < perPage) {
      return null;
    }

    page++;
  }
}

/**
 * Check if a comment is minimized
 */
async function getMinimizedState(nodeId) {
  if (!nodeId) return false;

  const query = `
    query($id: ID!) {
      node(id: $id) {
        ... on IssueComment {
          isMinimized
        }
      }
    }
  `;

  const result = await githubGraphQL(query, { id: nodeId });
  return Boolean(result?.data?.node?.isMinimized);
}

/**
 * Minimize a comment as outdated
 */
async function minimizeComment(nodeId) {
  const mutation = `
    mutation($id: ID!) {
      minimizeComment(input: { subjectId: $id, classifier: OUTDATED }) {
        minimizedComment {
          isMinimized
        }
      }
    }
  `;

  await githubGraphQL(mutation, { id: nodeId });
}

/**
 * Unminimize a comment
 */
async function unminimizeComment(nodeId) {
  const mutation = `
    mutation($id: ID!) {
      unminimizeComment(input: { subjectId: $id }) {
        unminimizedComment {
          isMinimized
        }
      }
    }
  `;

  await githubGraphQL(mutation, { id: nodeId });
}

/**
 * Create a new comment
 */
async function createComment(body) {
  return githubRequest(
    'POST',
    `/repos/${owner}/${repo}/issues/${prNumber}/comments`,
    { body }
  );
}

/**
 * Update an existing comment
 */
async function updateComment(commentId, body) {
  return githubRequest(
    'PATCH',
    `/repos/${owner}/${repo}/issues/comments/${commentId}`,
    { body }
  );
}

/**
 * Main logic
 */
async function main() {
  const existing = await findAlertComment();
  const needsAlert = alertLevel === 'warning' || alertLevel === 'error';

  if (needsAlert) {
    console.log(`Lockfile drift detected (level: ${alertLevel}), posting/updating alert...`);

    if (existing) {
      const isMinimized = await getMinimizedState(existing.node_id);
      if (isMinimized) {
        console.log('Unminimizing existing comment...');
        await unminimizeComment(existing.node_id);
      }
      console.log(`Updating comment ${existing.id}...`);
      await updateComment(existing.id, alertBody);
    } else {
      console.log('Creating new alert comment...');
      await createComment(alertBody);
    }
  } else {
    console.log('Lockfile is in sync (no alert needed).');

    if (existing) {
      const isMinimized = await getMinimizedState(existing.node_id);
      if (!isMinimized) {
        console.log(`Minimizing comment ${existing.id}...`);
        await minimizeComment(existing.node_id);
      } else {
        console.log('Alert comment already minimized.');
      }
    } else {
      console.log('No alert comment found.');
    }
  }
}

main().catch((error) => {
  console.error('Error:', error.message);
  process.exit(1);
});
