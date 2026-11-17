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
    // A search that resolves after the input has changed must not write its
    // result (#17108): the cleanup marks this run superseded, and the run
    // skips every state write — including the loading flag, which the run
    // for the new input now owns.
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

            if (superseded) return;
            setUrlSearchResults(repositories.items);
          } catch {
            if (superseded) return;
            setUrlSearchResults([]);
          } finally {
            if (!superseded) setIsUrlSearchLoading(false);
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
