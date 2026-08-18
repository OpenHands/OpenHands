import { useState, useEffect } from "react";
import { Provider } from "#/types/settings";
import { GitRepository } from "#/types/git";
import GitService from "#/api/git-service/git-service.api";

export function useUrlSearch(
  inputValue: string,
  provider: Provider | null | undefined,
) {
  const [urlSearchResults, setUrlSearchResults] = useState<GitRepository[]>([]);
  const [isUrlSearchLoading, setIsUrlSearchLoading] = useState(false);

  useEffect(() => {
    const handleUrlSearch = async () => {
      // Guard against null/undefined provider to prevent sending
      // requests via the cloud proxy before providers have loaded
      if (!provider) {
        setUrlSearchResults([]);
        return;
      }

      if (inputValue.startsWith("https://")) {
        const match = inputValue.match(/https:\/\/[^/]+\/([^/]+\/[^/]+)/);
        if (match) {
          const repoName = match[1];

          setIsUrlSearchLoading(true);
          try {
            const repositories = await GitService.searchGitRepositories(
              repoName,
              provider,
              3,
            );

            setUrlSearchResults(repositories.items);
          } catch {
            setUrlSearchResults([]);
          } finally {
            setIsUrlSearchLoading(false);
          }
        } else {
          // HTTPS URL that does not contain an owner/repository path.
          // Clear any prior results so the dropdown never shows a
          // result that no longer corresponds to the current input.
          setUrlSearchResults([]);
        }
      } else {
        setUrlSearchResults([]);
      }
    };

    handleUrlSearch();
  }, [inputValue, provider]);

  return { urlSearchResults, isUrlSearchLoading };
}
