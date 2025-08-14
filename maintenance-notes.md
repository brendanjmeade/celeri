# Maintenance notes

This document captures current best practices for maintaining this repository.

## Cutting a new release

- **Versioning is tag-driven**: The package version is derived from Git tags via `hatch-vcs`. Tags must match `vX.Y.Z` (see semantic versioning at [semantic versioning](https://semver.org/)).

1. Open [Releases](https://github.com/brendanjmeade/celeri/releases) and click [Draft a new release](https://github.com/brendanjmeade/celeri/releases/new).
2. Click "Tag: Select tag" and enter `vX.Y.Z` according to the desired version number. Click "+ Create new tag on publish" to accept the tag.
3. Click "Generate release notes" to prefill the title and notes.
4. Optional but recommended: Add a concise summary of the most important changes and curate the PR list into categories (e.g., "Bugfix", "New features", "Maintenance"). Remove irrelevant auto-generated entries.
5. Click "Publish release".
6. Go to [Actions](https://github.com/brendanjmeade/celeri/actions) and open the latest run of the `release-pipeline` workflow. After the build finishes, the publish step will wait for approval (shown with an orange clock icon). Click "Review deployments".
7. In "Review pending deployments", select the `release` environment and click "Approve and deploy".
8. The `publish-package` job will publish to PyPI. Once complete, the new version appears immediately on the [celeri PyPI page](https://pypi.org/project/celeri/).

### In case something goes wrong

#### Error or omission in GitHub release notes

You can edit release notes at any time, even after publishing.

#### Failure in the release workflow

- If it looks transient (e.g., temporary credentials), retry the job.
- If a code change is required, you must create a new release because releases are tied to specific commits.

If nothing has been published to PyPI yet, you can redo the same version:

1. Delete the failed GitHub release from [Releases](https://github.com/brendanjmeade/celeri/releases).
2. **IMPORTANT**: Delete the corresponding `vX.Y.Z` [tag](https://github.com/brendanjmeade/celeri/tags).
3. Start again from [Cutting a new release](#cutting-a-new-release).
