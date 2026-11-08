import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../test-utils";
import { ConversationHistorySearch } from "#/components/features/chat/conversation-history-search";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import EventService from "#/api/event-service/event-service.api";
import type { MessageEvent } from "#/types/agent-server/core";

const search = vi.spyOn(EventService, "searchEvents");
const count = vi.spyOn(EventService, "getEventCount");
const message: MessageEvent = {
  id: "old-match",
  timestamp: "2026-11-01T00:00:00Z",
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "old error decision" }],
  },
  activated_skills: [],
  extended_content: [],
};

function renderSearch(id = "conversation-a") {
  return renderWithProviders(
    <ActiveBackendProvider>
      <ConversationHistorySearch
        conversationId={id}
        conversationUrl="https://runtime.test"
        sessionApiKey="private"
      />
    </ActiveBackendProvider>,
  );
}

async function begin() {
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "HISTORY_SEARCH$OPEN" }));
  await user.type(screen.getByRole("searchbox"), "error");
  await user.click(
    screen.getByRole("button", { name: "HISTORY_SEARCH$SUBMIT" }),
  );
  return user;
}

describe("Find in conversation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setRegisteredBackends([
      {
        id: "local-a",
        kind: "local",
        name: "Local",
        host: "http://local.test",
        apiKey: "",
      },
      {
        id: "cloud-b",
        kind: "cloud",
        name: "Cloud",
        host: "https://cloud.test",
        apiKey: "key",
      },
    ]);
    setActiveSelection({ backendId: "local-a" });
    count.mockResolvedValue(1);
    search.mockResolvedValue({ items: [message] });
  });

  it("searches only after submit, opens bounded context and returns focus to live", async () => {
    const user = userEvent.setup();
    renderSearch();
    await user.click(
      screen.getByRole("button", { name: "HISTORY_SEARCH$OPEN" }),
    );
    await user.type(screen.getByRole("searchbox"), "error");
    expect(search).not.toHaveBeenCalled();
    await user.keyboard("{Enter}");
    await screen.findByText("HISTORY_SEARCH$COMPLETE");
    const result = screen.getByRole("button", { name: /old error decision/ });
    expect(result).toHaveTextContent("2026");
    await user.click(result);
    await screen.findByTestId("history-search-context");
    expect(search).toHaveBeenCalledTimes(2);
    expect(
      search.mock.calls.slice(1).every((call) => call[3]?.limit === 100),
    ).toBe(true);
    await user.click(
      screen.getByRole("button", { name: "HISTORY_SEARCH$RETURN_LIVE" }),
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "HISTORY_SEARCH$OPEN" }),
    ).toHaveFocus();
  });

  it("reports errors and retains an explicit retry path", async () => {
    search.mockRejectedValueOnce(new Error("private server diagnostic"));
    renderSearch();
    const user = await begin();
    await screen.findByRole("alert");
    expect(
      screen.queryByText("private server diagnostic"),
    ).not.toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "HISTORY_SEARCH$CONTINUE" }),
    );
    await screen.findByText("HISTORY_SEARCH$COMPLETE");
  });

  it("cancels a pending page and discards its late match", async () => {
    let resolve!: (page: { items: MessageEvent[] }) => void;
    search.mockImplementationOnce(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    renderSearch();
    const user = await begin();
    await waitFor(() => expect(search).toHaveBeenCalledTimes(1));
    await user.click(
      screen.getByRole("button", { name: "HISTORY_SEARCH$CANCEL" }),
    );
    await screen.findByText("HISTORY_SEARCH$CANCELLED");
    await act(async () => {
      resolve({ items: [message] });
    });
    expect(
      screen.queryByRole("button", { name: /old error decision/ }),
    ).not.toBeInTheDocument();
  });

  it.each(["backend", "organization", "credentials"])(
    "invalidates a pending read after a %s change",
    async (change) => {
      let resolve!: (page: { items: MessageEvent[] }) => void;
      search.mockImplementationOnce(
        () =>
          new Promise((done) => {
            resolve = done;
          }),
      );
      renderSearch();
      await begin();
      await waitFor(() => expect(search).toHaveBeenCalledTimes(1));
      await act(async () => {
        if (change === "credentials")
          setRegisteredBackends([
            {
              id: "local-a",
              kind: "local",
              name: "Local",
              host: "http://local.test",
              apiKey: "rotated",
              connectionRevision: 2,
            },
          ]);
        else
          setActiveSelection({
            backendId: change === "backend" ? "cloud-b" : "local-a",
            orgId: change === "organization" ? "org-new" : null,
          });
        resolve({ items: [message] });
      });
      expect(
        screen.queryByRole("button", { name: /old error decision/ }),
      ).not.toBeInTheDocument();
      expect(screen.getByRole("searchbox")).toHaveValue("");
    },
  );

  it("clears results on conversation changes", async () => {
    const view = renderSearch();
    await begin();
    await screen.findByRole("button", { name: /old error decision/ });
    view.rerender(
      <ActiveBackendProvider>
        <ConversationHistorySearch conversationId="conversation-b" />
      </ActiveBackendProvider>,
    );
    expect(
      screen.queryByRole("button", { name: /old error decision/ }),
    ).not.toBeInTheDocument();
  });

  it("works without an active Cloud runtime when persisted App API history is available", async () => {
    setActiveSelection({ backendId: "cloud-b", orgId: "org-one" });
    count.mockRejectedValue(new Error("archived runtime unavailable"));
    renderSearch();
    await begin();
    await screen.findByRole("button", { name: /old error decision/ });
    expect(search.mock.calls[0][3]).toMatchObject({
      strictPagination: true,
      limit: 100,
    });
  });

  it("shows incomplete progress after 500 events and requires explicit continuation", async () => {
    count.mockResolvedValue(501);
    search.mockImplementation(async (_id, _url, _key, options) => {
      const page = Number(options?.pageId ?? 0);
      if (page === 5) return { items: [message], next_page_id: null };
      return {
        items: Array.from({ length: 100 }, (_, n) => ({
          ...message,
          id: `page-${page}-${n}`,
          llm_message: {
            role: "user" as const,
            content: [{ type: "text" as const, text: "ordinary turn" }],
          },
        })),
        next_page_id: String(page + 1),
      };
    });
    renderSearch();
    const user = await begin();
    await screen.findByText("HISTORY_SEARCH$INCOMPLETE");
    expect(search).toHaveBeenCalledTimes(5);
    expect(
      screen.queryByRole("button", { name: /old error decision/ }),
    ).not.toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "HISTORY_SEARCH$CONTINUE" }),
    );
    await screen.findByText("HISTORY_SEARCH$COMPLETE");
    await screen.findByRole("button", { name: /old error decision/ });
    expect(search).toHaveBeenCalledTimes(6);
  });

  it("moves between result buttons by keyboard and opens context with Enter", async () => {
    const second = {
      ...message,
      id: "second-match",
      llm_message: {
        role: "assistant" as const,
        content: [{ type: "text" as const, text: "second error match" }],
      },
    };
    search.mockResolvedValue({ items: [message, second] });
    count.mockResolvedValue(2);
    renderSearch();
    const user = await begin();
    const first = await screen.findByRole("button", {
      name: /old error decision/,
    });
    const next = screen.getByRole("button", { name: /second error match/ });
    act(() => first.focus());
    await user.keyboard("{ArrowDown}");
    expect(next).toHaveFocus();
    await user.keyboard("{ArrowUp}");
    expect(first).toHaveFocus();
    await user.keyboard("{Enter}");
    await screen.findByTestId("history-search-context");
    expect(
      screen.getByRole("heading", { name: "HISTORY_SEARCH$CONTEXT" }),
    ).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("can cancel the count request and start again without displaying its late response", async () => {
    let resolve!: (count: number) => void;
    count.mockImplementationOnce(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    renderSearch();
    const user = await begin();
    await user.click(
      screen.getByRole("button", { name: "HISTORY_SEARCH$CANCEL" }),
    );
    await act(async () => {
      resolve(10000);
    });
    expect(search).not.toHaveBeenCalled();
    await user.click(
      screen.getByRole("button", { name: "HISTORY_SEARCH$CONTINUE" }),
    );
    await screen.findByRole("button", { name: /old error decision/ });
  });

  it("reports a context read failure and lets the same result retry", async () => {
    renderSearch();
    const user = await begin();
    const result = await screen.findByRole("button", {
      name: /old error decision/,
    });
    search.mockRejectedValueOnce(new Error("runtime unavailable"));
    await user.click(result);
    await screen.findByText("HISTORY_SEARCH$CONTEXT_ERROR");
    await user.click(result);
    await screen.findByTestId("history-search-context");
  });

  it("does not restore a previous query's matches when cancelling a new count read", async () => {
    renderSearch();
    const user = await begin();
    await screen.findByRole("button", { name: /old error decision/ });
    let resolve!: (total: number) => void;
    count.mockImplementationOnce(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    await user.clear(screen.getByRole("searchbox"));
    await user.type(screen.getByRole("searchbox"), "different query");
    await user.click(
      screen.getByRole("button", { name: "HISTORY_SEARCH$SUBMIT" }),
    );
    await user.click(
      screen.getByRole("button", { name: "HISTORY_SEARCH$CANCEL" }),
    );
    expect(
      screen.queryByRole("button", { name: /old error decision/ }),
    ).not.toBeInTheDocument();
    await act(async () => {
      resolve(1);
    });
  });
});
