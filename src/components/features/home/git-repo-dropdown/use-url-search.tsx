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
    // Set when a newer input value supersedes this run. Without it, a
    // request that resolves after the user has edited or cleared the field
    // still writes its results, listing a repository for text that is gone.
    let superseded = false;

    const handleUrlSearch = async () => {
      // Guard against null/undefined provider to prevent sending
      // requests via the cloud proxy before providers have loaded
      if (!provider) {
        setUrlSearchResults([]);
        setIsUrlSearchLoading(false);
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

            if (!superseded) {
              setUrlSearchResults(repositories.items);
            }
          } catch {
            if (!superseded) {
              setUrlSearchResults([]);
            }
          } finally {
            if (!superseded) {
              setIsUrlSearchLoading(false);
            }
          }
        } else {
          setUrlSearchResults([]);
          setIsUrlSearchLoading(false);
        }
      } else {
        setUrlSearchResults([]);
        setIsUrlSearchLoading(false);
      }
    };

    handleUrlSearch();

    return () => {
      superseded = true;
    };
  }, [inputValue, provider]);

  return { urlSearchResults, isUrlSearchLoading };
}
