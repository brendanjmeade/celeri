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

## Modifying dependencies

Dependency requirements are specified in both `pixi.toml` and `pyproject.toml`. The former defines the conda-forge requirements, while the latter defines the PyPI requirements. Usually package names are the same on conda-forge and PyPI, but occasionally they differ. The easiest way to search for conda-forge packages is <prefix.dev>, while PyPI packages are indexed on <pypi.org>.

Specific package versions that satisfy the requirements for each platform (Linux, macOS, Windows) are specified in `pixi.lock`. This file is managed by `pixi` and should never be manually edited. After updating **either `pixi.toml` or `pyproject.toml`, you must run `pixi install` to update both your environment and the lockfile**. Also, this lockfile **must be committed to Git** to ensure that the environment is reproducible, otherwise tests in PRs will fail. See [Committing the lockfile and avoiding merge conflicts](#committing-the-lockfile-and-avoiding-merge-conflicts) for more details.

### Updatinng existing dependencies

To update a single package to the latest version, either adjust the corresponding constraint in `pixi.toml` or run

```bash
pixi update <package-name>
```

To update all packages to their latest versions, run

```bash
pixi update
```

In case you are updating a package due to an incompatibility, it's recommended to adjust the corresponding lower version bound in `pixi.toml` and `pyproject.toml` so that incompatible versions are excluded.

### Adding new dependencies

To add a new dependency to the project, add the package to `pixi.toml` and run `pixi install` to install it and update the lockfile. If the package is also a core dependency, it should also be added to `pyproject.toml`. (See [What to put in `pixi.toml` vs `pyproject.toml`](#what-to-put-in-pixitoml-vs-pyprojecttoml) for more details.)

Alternatively, there is a helper command `pixi add <some-package>` that will add the package to `pixi.toml` and automatically run `pixi install`.

### Committing the lockfile and avoiding merge conflicts

Anytime you commit an update to either `pixi.toml` or `pyproject.toml`, it is essential to also commit an updated lockfile to Git.

Because lockfile merge conflicts are common, it's strongly encouraged to put the lockfile update into a separate commit to make it easier to revert if necessary.

A merge conflict in `pixi.lock` will occur anytime two branches update the lockfile. Assuming you need to merge your local branch into the remote main branch, the simplest procedure is:

1. Ensure that your local copy of the `main` branch is up to date

   ```bash
   git pull upstream main:main
   ```

   (If you are a core maintainer working directly on `brendanjmeade/celeri` rather than a fork, then replace `upstream` by `origin`.)

2. Replace your branch's `pixi.lock` with the `main` branch's `pixi.lock`.

   ```bash
   git checkout main -- pixi.lock
   ```

3. Commit the change.

   ```bash
   git commit -m "Replace pixi.lock with main branch"
   ```

4. Merge `main` into your branch.

   ```bash
   git merge main
   ```

5. Resolve any remaining merge conflicts, if any. (The lockfile should no longer be a conflict.)

6. Push the changes to your remote branch.

   ```bash
   git push
   ```

It's also possible to resolve the merge conflict with a rebase, but this approach is more advanced and prone to complications.

### What to put in `pixi.toml` vs `pyproject.toml`

The `pyproject.toml` file defines the requirements of the Python package. The `pixi.toml` includes the Python package as a dependency. Therefore, dependencies in `pixi.toml` should be a superset of the dependencies in `pyproject.toml`. The `pixi.toml` should include all the development dependencies, notebook dependencies, and all other convenience packages. In contrast, `pyproject.toml` can be more slimmed down. Any dependency deemed to be "optional" must be excluded from `pyproject.toml`, since it is impossible to install `celeri` in a (consistent) Python environment without also installing all the packages in `pyproject.toml`.
