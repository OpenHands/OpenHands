import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { InsiderCatBadge } from "./insider-cat-badge";

describe("Insider Cat identity", () => {
  it.each<{
    tags: Record<string, string> | null;
    parent: string | null;
    expected: boolean;
  }>([
    { tags: { smolpaws: "insider" }, parent: null, expected: true },
    {
      tags: { smolpaws: "insider", insiderrole: "controller" },
      parent: null,
      expected: true,
    },
    {
      tags: { smolpaws: "insider", insiderrole: "worker" },
      parent: null,
      expected: false,
    },
    {
      tags: { smolpaws: "insider" },
      parent: "22222222-2222-4222-8222-222222222222",
      expected: false,
    },
    { tags: null, parent: null, expected: false },
  ])(
    "marks only a top-level current or legacy controller: %j",
    ({ tags, parent, expected }) => {
      renderWithProviders(
        <InsiderCatBadge tags={tags} parentConversationId={parent} />,
      );
      expect(Boolean(screen.queryByTestId("insider-cat-badge"))).toBe(expected);
    },
  );

  it("returns to the same full controller ID without creating a conversation", async () => {
    const navigate = vi.fn();
    const id = "22222222-2222-4222-8222-222222222222";
    renderWithProviders(
      <InsiderCatBadge
        tags={{ smolpaws: "insider", insiderrole: "controller" }}
        conversationId={id}
      />,
      { navigation: { navigate } },
    );

    await userEvent.click(
      screen.getByRole("link", { name: "CONVERSATION$OPEN_INSIDER_CAT" }),
    );

    const [path] = navigate.mock.calls[0];
    expect(path.split("?")[0]).toBe(
      `/extensions/insider-cat/projects/conversations/${id}`,
    );
  });
});
