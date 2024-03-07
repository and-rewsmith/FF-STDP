# Contributing to this project

We appreciate your help in improving and maintaining this project. Here's a quick guide on how you can help:

## PR Title Format

All PRs should be titled in the following format: [REASON]: [DESCRIPTION]

Here's what you need to know about these placeholders:

- `[REASON]` must be one of the following:

    - `feat`: New feature for the user, not a new feature for a build script
    - `fix`: Bug fix for the user, not a fix to a build script
    - `docs`: Documentation only changes
    - `style`: Formatting, missing semicolons, etc; no production code change
    - `refactor`: Refactoring production code, eg. renaming a variable
    - `chore`: Updating grunt tasks etc; no production code change
    - `perf`: A code change that improves performance
    - `test`: Adding missing tests, refactoring tests; no production code change
    - `build`: Changes that affect the build system or external dependencies (example scopes: Jenkins, Makefile)
    - `ci`: Changes provided by DevOps for CI purposes
    - `revert`: Reverts a previous commit


- `[DESCRIPTION]` should be a concise description of the changes in the PR.

Examples of correctly formatted titles:

- `[feat]: add search feature to home page`
- `[fix]: correct typo in introduction paragraph`
- `[docs]: update the README with new information`
- `[style]: format code according to the style guide`
- `[refactor]: change variable names for clarity`
- `[chore]: update build scripts`
- `[perf]: improve load times`
- `[test]: add tests for new feature`
- `[build]: update dependencies`
- `[ci]: adjust build process for new CI server`
- `[revert]: revert commit abcd1234`

We have a GitHub Action in place that will check your PR title for this format when you open or update a PR. If the title does not match the expected format, the check will fail and you will need to update the title.
