const KEY_1 = "INCREASE_TEST_COVERAGE";
const VALUE_1 = `I want to increase the test coverage of the repository in the current directory.

Please investigate the repo to figure out what language is being used, and where tests are located, if there are any.

If there are no tests already in the repo, add a very basic test, using typical testing strategies for the language involved.

If there are existing tests, find a function or method which lacks adequate unit tests, and add unit tests for it. Be sure to respect the projects existing test structures.

Make sure the tests pass before you finish.`;

const KEY_2 = "AUTO_MERGE_PRS";
const VALUE_2 = `Please add a GitHub action to this repository which automatically merges pull requests from Dependabot so long as the tests are passing.`;

const KEY_3 = "FIX_README";
const VALUE_3 = `Please look at the README and make the following improvements, if they make sense:
* correct any typos that you find
* add missing language annotations on codeblocks
* if there are references to other files or other sections of the README, turn them into links
* make sure the readme has an h1 title towards the top
* make sure any existing sections in the readme are appropriately separated with headings

If there are no obvious ways to improve the README, make at least one small change to make the wording clearer or friendlier`;

const KEY_4 = "CLEAN_DEPENDENCIES";
const VALUE_4 = `Examine the dependencies of the current codebase. Make sure you can run the code and any tests.

Then run any commands necessary to update all dependencies to the latest versions, and make sure the code continues to run correctly and the tests pass. If changes need to be made to the codebase, go ahead and make those changes. You can look up documentation for new versions using the browser if you need to.

If a particular dependency update is causing trouble (e.g. breaking changes that you can't fix), you can revert it and send a message to the user explaining why.

Additionally, if you're able to prune any dependencies that are obviously unused, please do so. You may use third party tools to check for unused dependencies.`;

const KEY_5 = "ADD_DOCS";
const VALUE_5 = `Investigate the documentation in the root of the current repo. Please add a CODE_OF_CONDUCT.md and CONTRIBUTORS.md with good defaults if they are not present. Use information in the README to inform the CONTRIBUTORS doc. If there is no LICENSE currently in the repo, please add the Apache 2.0 license. Add links to all these documents into the README`;

const KEY_6 = "ADD_DOCKERFILE";
const VALUE_6 = `Investigate the current repo to understand the installation instructions. Then create a Dockerfile that runs the application, using best practices like arguments and multi-stage builds wherever appropriate.

If there is an existing Dockerfile, and there are ways to improve it according to best practices, do so.`;

export const REPO_SUGGESTIONS: Record<string, string> = {
  [KEY_1]: VALUE_1,
  [KEY_2]: VALUE_2,
  [KEY_3]: VALUE_3,
  [KEY_4]: VALUE_4,
  [KEY_5]: VALUE_5,
  [KEY_6]: VALUE_6,
};
