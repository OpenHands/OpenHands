import { HttpError } from "@openhands/typescript-client";
import { afterEach, describe, expect, it, vi } from "vitest";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import ProjectsService from "#/api/projects-service/projects-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { setResolved } from "#/components/features/projects/project-board";

describe("project board resolve and reopen", () => {
  afterEach(() => vi.restoreAllMocks());

  it.each([true, false])(
    "uses the project's route (resolved=%s) and leaves the tags alone",
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

  it("falls back to fresh tags on a server without the route", async () => {
    vi.spyOn(ProjectsService, "setConversationResolved").mockRejectedValue(
      new HttpError(404, "Not Found", { detail: "Not Found" }),
    );
    const current = {
      id: "c1",
      tags: { project: "p1", resolvenote: "0.9 critic", team: "web" },
    } as unknown as AppConversation;
    vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    ).mockResolvedValue([current]);
    const replace = vi
      .spyOn(AgentServerConversationService, "replaceConversationTags")
      .mockResolvedValue(undefined as never);

    await setResolved("p1", "c1", true);

    expect(replace).toHaveBeenCalledWith("c1", current.tags, {
      project: "p1",
      team: "web",
      resolved: "user",
    });
  });

  it("reports any other failure without touching the tags", async () => {
    vi.spyOn(ProjectsService, "setConversationResolved").mockRejectedValue(
      new HttpError(500, "Server Error", {}),
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
