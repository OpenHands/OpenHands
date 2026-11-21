import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";
import {
  useRenameWorkspace,
  useRenameWorkspaceParent,
} from "#/hooks/mutation/use-local-workspaces-mutations";
import { LOCAL_WORKSPACES_QUERY_KEYS } from "#/hooks/query/query-keys";

const { displayErrorToast, displaySuccessToast } = vi.hoisted(() => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast,
  displaySuccessToast,
}));

vi.mock("#/api/workspaces-service/workspaces-service.api", () => ({
  default: {
    renameWorkspace: vi.fn(),
    renameWorkspaceParent: vi.fn(),
  },
}));

const WORKSPACES = [
  { id: "/a", name: "a", path: "/a" },
  { id: "/b", name: "b", path: "/b" },
];

const PARENTS = [{ id: "/p", name: "p", path: "/p" }];

function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
}

function seed(queryClient: QueryClient) {
  queryClient.setQueryData(LOCAL_WORKSPACES_QUERY_KEYS.all, {
    workspaces: WORKSPACES,
    workspaceParents: PARENTS,
  });
}

function read(queryClient: QueryClient) {
  return queryClient.getQueryData<{
    workspaces: { id: string; name: string; path: string }[];
    workspaceParents: { id: string; name: string; path: string }[];
  }>(LOCAL_WORKSPACES_QUERY_KEYS.all);
}

type RenameHook = () => {
  mutate: (variables: { path: string; name: string }) => void;
  mutateAsync: (variables: { path: string; name: string }) => Promise<unknown>;
};

function renderRenameHook(hook: RenameHook) {
  const queryClient = createQueryClient();
  seed(queryClient);
  const utils = renderHook(hook, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
  return { queryClient, ...utils };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("useRenameWorkspace", () => {
  it("rewrites the label optimistically while the request is in flight", async () => {
    let resolveRename: () => void = () => {};
    vi.mocked(WorkspacesService.renameWorkspace).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRename = () =>
            resolve({ workspaces: WORKSPACES, workspaceParents: PARENTS });
        }),
    );

    const { queryClient, result } = renderRenameHook(useRenameWorkspace);

    await act(async () => {
      result.current.mutate({ path: "/a", name: "renamed" });
    });

    const cached = read(queryClient);
    expect(cached?.workspaces[0]).toEqual({
      id: "/a",
      name: "renamed",
      path: "/a",
    });
    expect(cached?.workspaces[1].name).toBe("b");

    await act(async () => {
      resolveRename();
    });
  });

  it("rolls back and toasts when the rename fails", async () => {
    vi.mocked(WorkspacesService.renameWorkspace).mockRejectedValue(
      new Error("boom"),
    );

    const { queryClient, result } = renderRenameHook(useRenameWorkspace);

    await act(async () => {
      await expect(
        result.current.mutateAsync({ path: "/a", name: "renamed" }),
      ).rejects.toThrow("boom");
    });

    expect(read(queryClient)?.workspaces[0].name).toBe("a");
    expect(displayErrorToast).toHaveBeenCalledTimes(1);
  });

  it("refetches the list once the rename settles", async () => {
    vi.mocked(WorkspacesService.renameWorkspace).mockResolvedValue({
      workspaces: WORKSPACES,
      workspaceParents: PARENTS,
    });

    const { queryClient, result } = renderRenameHook(useRenameWorkspace);
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    await act(async () => {
      await result.current.mutateAsync({ path: "/a", name: "renamed" });
    });

    await waitFor(() =>
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: LOCAL_WORKSPACES_QUERY_KEYS.all,
      }),
    );
    expect(displaySuccessToast).toHaveBeenCalledTimes(1);
  });
});

describe("useRenameWorkspaceParent", () => {
  it("only touches the parent entry", async () => {
    vi.mocked(WorkspacesService.renameWorkspaceParent).mockResolvedValue({
      workspaces: WORKSPACES,
      workspaceParents: PARENTS,
    });

    const { queryClient, result } = renderRenameHook(useRenameWorkspaceParent);

    await act(async () => {
      await result.current.mutateAsync({ path: "/p", name: "renamed" });
    });

    expect(WorkspacesService.renameWorkspaceParent).toHaveBeenCalledWith(
      "/p",
      "renamed",
    );
    expect(WorkspacesService.renameWorkspace).not.toHaveBeenCalled();
    expect(read(queryClient)?.workspaces[0].name).toBe("a");
  });
});
