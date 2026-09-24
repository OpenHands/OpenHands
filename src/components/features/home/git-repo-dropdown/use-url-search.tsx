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
    let isCurrent = true;

    const handleUrlSearch = async () => {
      // Guard against null/undefined provider to prevent sending
      // requests via the cloud proxy before providers have loaded
      if (!provider) {
        setUrlSearchResults([]);
        setIsUrlSearchLoading(false);
        return;
      }

      if (!inputValue.startsWith("https://")) {
        setUrlSearchResults([]);
        setIsUrlSearchLoading(false);
        return;
      }

      const match = inputValue.match(/https:\/\/[^/]+\/([^/]+\/[^/]+)/);
      if (!match) {
        setUrlSearchResults([]);
        setIsUrlSearchLoading(false);
        return;
      }

      const repoName = match[1];

      setIsUrlSearchLoading(true);
      try {
        const repositories = await GitService.searchGitRepositories(
          repoName,
          provider,
          3,
        );

        if (isCurrent) {
          setUrlSearchResults(repositories.items);
        }
      } catch {
        if (isCurrent) {
          setUrlSearchResults([]);
        }
      } finally {
        if (isCurrent) {
          setIsUrlSearchLoading(false);
        }
      }
    };

    handleUrlSearch();

    return () => {
      isCurrent = false;
    };
  }, [inputValue, provider]);

  return { urlSearchResults, isUrlSearchLoading };
}
