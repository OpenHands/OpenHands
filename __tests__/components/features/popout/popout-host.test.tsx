import { fireEvent, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { PopoutHost } from "#/components/features/popout/popout-host";
import { usePopoutStore } from "#/stores/popout-store";
import { I18nKey } from "#/i18n/declaration";

const navigateMock = vi.fn();

vi.mock("#/context/navigation-context", async (importActual) => ({
  ...(await importActual<object>()),
  useNavigation: () => ({
    currentPath: "/conversations/source",
    conversationId: "source",
    isNavigating: false,
    navigate: navigateMock,
  }),
}));

// The host mounts a live conversation per window; stub it so this test stays
// focused on chrome actions (maximize / close / minimize).
vi.mock("#/components/features/popout/popout-conversation", () => ({
  PopoutConversation: ({ conversationId }: { conversationId: string }) => (
    <div data-testid={`fork-conversation-${conversationId}`} />
  ),
}));

describe("PopoutHost", () => {
  beforeEach(() => {
    navigateMock.mockReset();
    usePopoutStore.setState({ popouts: [] });
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: 1280,
    });
  });

  it("renders open popouts and maximizes by navigating then closing", () => {
    usePopoutStore.getState().openPopout({
      conversationId: "fork-1",
      title: "Trip planning (branch)",
    });

    renderWithProviders(<PopoutHost />);

    expect(screen.getByTestId("popout-host")).toBeInTheDocument();
    expect(screen.getByTestId("popout-fork-1")).toBeInTheDocument();
    expect(screen.getByTestId("fork-conversation-fork-1")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: I18nKey.POPOUT$MAXIMIZE }),
    );

    expect(navigateMock).toHaveBeenCalledWith("/conversations/fork-1");
    expect(usePopoutStore.getState().popouts).toEqual([]);
  });

  it("closes a window from the title-bar control", () => {
    usePopoutStore.getState().openPopout({
      conversationId: "fork-1",
      title: "A",
    });

    renderWithProviders(<PopoutHost />);
    fireEvent.click(screen.getByRole("button", { name: I18nKey.POPOUT$CLOSE }));
    expect(usePopoutStore.getState().popouts).toEqual([]);
    expect(screen.queryByTestId("popout-host")).not.toBeInTheDocument();
  });

  it("moves overflowed popouts into a selector and promotes the selected one", () => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: 460,
    });
    usePopoutStore.setState({
      popouts: [
        {
          conversationId: "oldest",
          title: "Oldest",
          prefillMessage: null,
          mode: "expanded",
          openedAt: 1,
        },
        {
          conversationId: "middle",
          title: "Middle",
          prefillMessage: null,
          mode: "expanded",
          openedAt: 2,
        },
        {
          conversationId: "newest",
          title: "Newest",
          prefillMessage: null,
          mode: "expanded",
          openedAt: 3,
        },
      ],
    });

    renderWithProviders(<PopoutHost />);

    expect(screen.getByTestId("fork-conversation-newest")).toBeInTheDocument();
    expect(
      screen.queryByTestId("fork-conversation-oldest"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("popout-overflow-trigger")).toHaveTextContent("2");

    fireEvent.click(screen.getByTestId("popout-overflow-trigger"));
    const menu = screen.getByTestId("popout-overflow-menu");
    expect(within(menu).getByText("Oldest")).toBeInTheDocument();
    expect(within(menu).getByText("Middle")).toBeInTheDocument();
    expect(
      screen.getByTestId("popout-overflow-item-middle"),
    ).toHaveFocus();

    fireEvent.keyDown(menu, { key: "Escape" });
    expect(screen.queryByTestId("popout-overflow-menu")).not.toBeInTheDocument();
    expect(screen.getByTestId("popout-overflow-trigger")).toHaveFocus();

    fireEvent.click(screen.getByTestId("popout-overflow-trigger"));
    const reopenedMenu = screen.getByTestId("popout-overflow-menu");
    fireEvent.click(within(reopenedMenu).getByText("Oldest"));

    expect(screen.getByTestId("fork-conversation-oldest")).toBeInTheDocument();
    expect(
      screen.queryByTestId("fork-conversation-newest"),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("popout-overflow-menu")).not.toBeInTheDocument();
  });
});
