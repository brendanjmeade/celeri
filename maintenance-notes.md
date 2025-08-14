# Maintenance notes

This is a reference to document current best practices surrounding the maintenance of this repository.

## Cutting a new release

1. Under the ["Releases"](https://github.com/brendanjmeade/celeri/releases) element on the [main page](https://github.com/brendanjmeade/celeri), click the [Draft a new release](https://github.com/brendanjmeade/celeri/releases/new) button.
2. Click "Tag: Select tag" and type `vX.Y.Z` corresponding to the new version number. For recommended guidelines on choosing the next number, see [semantic versioning](https://semver.org/). Click "+ Create new tag on publish" to accept the tag.
3. Click the "Generate release notes" button. This will automatically populate "Release title" and "Release notes".
4. Optional but recommended: Give a summary of a sentence or two of the most important changes motivating the release. Also, curate the list PRs into categories, e.g. "Bugfix", "New features", "Maintenance", and delete any irrelevant machine-generated PRs.
5. Click "Publish release".
6. Go to the ["Actions"](https://github.com/brendanjmeade/celeri/actions) tab. There you will see a release workflow. After a short time the package will build automatically and await approval for release. This is indicated by a non-spinning orange clock icon. Click on the release workflow, and once it's awaiting approval, click "Review deployments".
7. In the "Review pending deployments" dialog, click the "release" checkbox, and click the "Approve and deploy" button.
8. The "publish-package" job will run and publish to PyPI. Once the job completes, the new version should be available instantaneously on the [celeri PyPI page](https://pypi.org/project/celeri/).

### In case something goes wrong

#### Error or omission in the GitHub release notes

No sweat, you can freely edit these at any time, even after publishing.

#### Failure in the release workflow

In case of a transitory failure, like expired credentials, you can safely retry the job.

If retrying doesn't help (usually it doesn't), then fixing the issue will require further commits, and therefore you MUST perform another release. This is essential, because the release is tied to the current Git commit.

Rather than create a new distinct version number, you can delete the existing failed release. This procedure consists of two steps:

1. Delete the failed GitHub release from ["Releases"](https://github.com/brendanjmeade/celeri/releases).
2. Delete the [tag](https://github.com/brendanjmeade/celeri/tags) `vX.Y.Z` corresponding to the failed release.

When deleting the release, **DON'T FORGET TO DELETE THE TAG**, otherwise things will go haywire.

Now you can cleanly start over with the steps for [Cutting a new release](#cutting-a-new-release).
