import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  buildWorkspaceUploadPath,
  getSafeUploadFileName,
  resolveAbsoluteWorkspacePath,
  resolveConversationUploadWorkingDir,
} from "#/api/workspace-upload-path";
import { clearAgentServerHomeDirCache } from "#/api/agent-server-home";

const mockGetHome = vi.fn();

vi.mock("@openhands/typescript-client/clients", () => ({
  FileClient: vi.fn(function FileClientMock() {
    return { getHome: mockGetHome };
  }),
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: vi.fn(() => ({
    host: "http://localhost:8000",
    apiKey: "test-key",
    workingDir: "workspace/project",
  })),
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      resolveConversationWorkingDir: vi.fn(
        async (id: string) => `/workspace/project/${id.replace(/-/g, "")}`,
      ),
    },
  }),
);

beforeEach(() => {
  clearAgentServerHomeDirCache();
  mockGetHome.mockReset();
  mockGetHome.mockResolvedValue({ home: "/Users/test" });
});

describe("workspace-upload-path", () => {
  // @spec WUP-001 — resolver anchors relative paths against /api/file/home.
  it("resolveAbsoluteWorkspacePath joins relative dirs to the agent-server home", async () => {
    const resolved = await resolveAbsoluteWorkspacePath("workspace/project");
    expect(resolved).toBe("/Users/test/workspace/project");
    expect(mockGetHome).toHaveBeenCalledTimes(1);
  });

  // @spec WUP-001 — absolute inputs pass through, no /file/home round-trip.
  it("resolveAbsoluteWorkspacePath leaves absolute paths alone", async () => {
    const resolved = await resolveAbsoluteWorkspacePath(
      "/workspace/project/custom",
    );
    expect(resolved).toBe("/workspace/project/custom");
    expect(mockGetHome).not.toHaveBeenCalled();
  });

  // @spec WUP-001 — Windows-style absolute paths are also pass-through.
  it("resolveAbsoluteWorkspacePath treats Windows drive-letter paths as absolute", async () => {
    const resolved = await resolveAbsoluteWorkspacePath("C:\\foo\\bar");
    expect(resolved).toBe("C:\\foo\\bar");
    expect(mockGetHome).not.toHaveBeenCalled();
  });

  // @spec WUP-001 — the home dir is cached so concurrent uploads share one round-trip.
  it("caches the home directory across calls", async () => {
    await resolveAbsoluteWorkspacePath("workspace/project");
    await resolveAbsoluteWorkspacePath("other/relative");
    expect(mockGetHome).toHaveBeenCalledTimes(1);
  });

  // @spec WUP-001 — failures are not cached so a later call retries fresh.
  it("does not cache failed lookups", async () => {
    mockGetHome.mockRejectedValueOnce(new Error("boom"));
    await expect(
      resolveAbsoluteWorkspacePath("workspace/project"),
    ).rejects.toThrow("boom");

    mockGetHome.mockResolvedValueOnce({ home: "/Users/test" });
    const resolved = await resolveAbsoluteWorkspacePath("workspace/project");
    expect(resolved).toBe("/Users/test/workspace/project");
    expect(mockGetHome).toHaveBeenCalledTimes(2);
  });

  // @spec WUP-001 — empty or whitespace working dirs collapse to the home dir.
  it("resolves an empty working dir to the home directory itself", async () => {
    expect(await resolveAbsoluteWorkspacePath("")).toBe("/Users/test");
    expect(await resolveAbsoluteWorkspacePath("/")).toBe("/Users/test");
  });

  // @spec WUP-001 — the safe-name helper still strips traversal segments.
  it("strips path segments from file names", () => {
    expect(getSafeUploadFileName("../../evil.txt")).toBe("evil.txt");
  });

  // @spec WUP-001 — buildWorkspaceUploadPath uses the resolver and a safe leaf.
  it("builds an absolute upload path from a relative working dir", async () => {
    const upload = await buildWorkspaceUploadPath("a.txt", "workspace/project");
    expect(upload).toBe("/Users/test/workspace/project/a.txt");
  });

  it("rejects file names that escape the destination via path traversal", async () => {
    const upload = await buildWorkspaceUploadPath(
      "../../evil.txt",
      "/workspace/project",
    );
    expect(upload).toBe("/workspace/project/evil.txt");
  });

  it("collapses trailing slashes on the working dir", async () => {
    const upload = await buildWorkspaceUploadPath(
      "doc.md",
      "/workspace/project/",
    );
    expect(upload).toBe("/workspace/project/doc.md");
  });

  it("prefers the active conversation workspace when ids match", async () => {
    const dir = await resolveConversationUploadWorkingDir("conv-uuid", {
      id: "conv-uuid",
      workspace: { working_dir: "/workspace/project/custom" },
    } as never);

    expect(dir).toBe("/workspace/project/custom");
  });

  it("resolves per-conversation dirs for UUID ids", async () => {
    const dir = await resolveConversationUploadWorkingDir(
      "550e8400-e29b-41d4-a716-446655440000",
      null,
    );

    expect(dir).toBe("/workspace/project/550e8400e29b41d4a716446655440000");
  });
});
