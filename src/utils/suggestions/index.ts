import { NON_REPO_SUGGESTIONS } from "./non-repo-suggestions";
import { REPO_SUGGESTIONS } from "./repo-suggestions";

export const SUGGESTIONS: Record<
  "repo" | "non-repo",
  Record<string, string>
> = {
  repo: REPO_SUGGESTIONS,
  "non-repo": NON_REPO_SUGGESTIONS,
};
