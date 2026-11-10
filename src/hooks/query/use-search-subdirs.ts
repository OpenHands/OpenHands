import { useQuery } from "@tanstack/react-query";
import { FileClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { useActiveBackend } from "#/contexts/active-backend-context";

type FileSubdirectoryPage = Awaited<
  ReturnType<FileClient["searchSubdirectories"]>
>;
type FileClientLike = Pick<FileClient, "searchSubdirectories">;

export interface FileBrowserEntry {
  label: string;
  path: string;
}

export interface HomeDirectoryResponse {
  home: string;
  favorites?: FileBrowserEntry[];
  locations?: FileBrowserEntry[];
}

function getFileClient() {
  return new FileClient(getAgentServerClientOptions());
}

export async function searchAllSubdirectories(
  path: string,
  fileClient: FileClientLike = getFileClient(),
): Promise<FileSubdirectoryPage> {
  const items: FileSubdirectoryPage["items"] = [];
  const seenPageIds = new Set<string>();
  let pageId: string | null | undefined;

  while (true) {
    const page = await fileClient.searchSubdirectories(
      path,
      pageId ? { pageId } : undefined,
    );
    items.push(...page.items);
    pageId = page.next_page_id;

    if (!pageId) {
      return { items, next_page_id: null };
    }

    if (seenPageIds.has(pageId)) {
      throw new Error("File search returned a repeated page id");
    }
    seenPageIds.add(pageId);
  }
}

export const useSearchSubdirs = (path: string | null) => {
  const active = useActiveBackend();
  return useQuery({
    queryKey: [
      "file",
      "search_subdirs",
      path,
      active.backend.id,
      active.orgId,
      // Editing a saved connection's host or API key keeps the backend id and
      // organization but points at a different server, so the revision has to
      // be part of the key — otherwise the previous connection's listing is
      // served from cache.
      active.backend.connectionRevision ?? 0,
    ],
    queryFn: () => searchAllSubdirectories(path as string),
    enabled: !!path,
    retry: false,
    meta: { disableToast: true },
  });
};

export const useHomeDirectory = () => {
  const active = useActiveBackend();
  return useQuery({
    queryKey: [
      "file",
      "home",
      active.backend.id,
      active.orgId,
      // HOME is cached forever (`staleTime: Infinity` below), so it must be
      // scoped to the connection revision for the same reason.
      active.backend.connectionRevision ?? 0,
    ],
    queryFn: async (): Promise<HomeDirectoryResponse> =>
      getFileClient().getHome(),
    retry: false,
    meta: { disableToast: true },
    staleTime: Infinity,
  });
};
