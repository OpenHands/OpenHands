import { HttpError } from "@openhands/typescript-client";
import { afterEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import ProjectsService from "#/api/projects-service/projects-service.api";
import { setResolved } from "#/components/features/projects/project-board";

describe("project board resolve and reopen", () => {
  afterEach(() => vi.restoreAllMocks());

  it.each([true, false])(
    "uses the project's route (resolved=%s) and never writes tags",
    async (resolved) => {
      const route = vi
        .spyOn(ProjectsService, "setConversationResolved")
        .mockResolvedValue(undefined);
      const replace = vi.spyOn(
        AgentServerConversationService,
        "replaceConversationTags",
      );

      await setResolved("p1", "c1", resolved);

      expect(route).toHaveBeenCalledWith("p1", "c1", resolved);
      expect(replace).not.toHaveBeenCalled();
    },
  );

  it("reports a refused route without touching the tags", async () => {
    vi.spyOn(ProjectsService, "setConversationResolved").mockRejectedValue(
      new HttpError(404, "Not Found", { detail: "Not Found" }),
    );
    const replace = vi.spyOn(
      AgentServerConversationService,
      "replaceConversationTags",
    );

    await expect(setResolved("p1", "c1", false)).rejects.toBeInstanceOf(
      HttpError,
    );
    expect(replace).not.toHaveBeenCalled();
  });
});
