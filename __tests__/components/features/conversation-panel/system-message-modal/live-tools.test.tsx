import { http, HttpResponse } from "msw";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import { SystemMessageModal } from "#/components/features/conversation-panel/system-message-modal";
import { server } from "#/mocks/node";
import { I18nKey } from "#/i18n/declaration";

const systemMessage = {
  content: "Original system prompt",
  tools: [],
  agent_class: null,
  openhands_version: null,
};
const conversationId = "11111111-1111-4111-8111-111111111111";

describe("conversation live tools", () => {
  it("shows live conversation tools with their source and reloads on reopening", async () => {
    const user = userEvent.setup();
    let reads = 0;
    server.use(
      http.get("*/server_info", () =>
        HttpResponse.json({ capabilities: ["conversation_live_tools_v1"] }),
      ),
      http.get("*/api/conversations/:id/tools", ({ params }) => {
        expect(params.id).toBe(conversationId);
        reads += 1;
        return HttpResponse.json({
          tools: [
            { name: "bash", description: "Run bash", source: "builtin" },
            {
              name: "global_search",
              description: "Search",
              source: "global",
              server_name: "global",
            },
            ...(reads > 1
              ? [
                  {
                    name: "scoped_fetch",
                    description: "Fetch",
                    source: "conversation",
                    server_name: "scoped",
                  },
                ]
              : []),
          ],
        });
      }),
    );

    const { rerender } = renderWithProviders(
      <SystemMessageModal
        isOpen
        conversationId={conversationId}
        systemMessage={systemMessage}
        onClose={() => {}}
      />,
    );
    await user.click(
      await screen.findByRole("tab", {
        name: I18nKey.SYSTEM_MESSAGE_MODAL$TOOLS_TAB,
      }),
    );
    expect(await screen.findByText("global_search")).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.SYSTEM_MESSAGE_MODAL$GLOBAL_TOOL),
    ).toBeInTheDocument();
    expect(screen.queryByText("scoped_fetch")).not.toBeInTheDocument();

    rerender(
      <SystemMessageModal
        isOpen={false}
        conversationId={conversationId}
        systemMessage={systemMessage}
        onClose={() => {}}
      />,
    );
    rerender(
      <SystemMessageModal
        isOpen
        conversationId={conversationId}
        systemMessage={systemMessage}
        onClose={() => {}}
      />,
    );
    await waitFor(() => expect(reads).toBe(2));
    expect(screen.getByText("scoped_fetch")).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.SYSTEM_MESSAGE_MODAL$CONVERSATION_ONLY_TOOL),
    ).toBeInTheDocument();
  });

  it("does not present launch-time tools as live when the inventory fails", async () => {
    const user = userEvent.setup();
    server.use(
      http.get("*/server_info", () =>
        HttpResponse.json({ capabilities: ["conversation_live_tools_v1"] }),
      ),
      http.get("*/api/conversations/:id/tools", () =>
        HttpResponse.json({ detail: "not available" }, { status: 503 }),
      ),
    );
    renderWithProviders(
      <SystemMessageModal
        isOpen
        conversationId={conversationId}
        systemMessage={{ ...systemMessage, tools: [{ name: "stale_tool" }] }}
        onClose={() => {}}
      />,
    );

    await user.click(
      await screen.findByRole("tab", {
        name: I18nKey.SYSTEM_MESSAGE_MODAL$TOOLS_TAB,
      }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      I18nKey.SYSTEM_MESSAGE_MODAL$TOOLS_LOAD_FAILED,
    );
    expect(screen.queryByText("stale_tool")).not.toBeInTheDocument();
  });
});
