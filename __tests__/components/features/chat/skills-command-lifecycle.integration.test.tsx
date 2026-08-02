import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useUiCommandInterceptor } from "#/hooks/chat/use-cli-command-interceptor";
import { SlashCommandMessages } from "#/components/features/chat/slash-command-messages";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";

const mocks = vi.hoisted(() => ({
  getLoadedResources: vi.fn(),
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { getLoadedResources: mocks.getLoadedResources },
  }),
);
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: { kind: "cloud" } }),
}));
vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({ navigate: vi.fn() }),
}));
vi.mock("#/hooks/use-breakpoint", () => ({
  SIDEBAR_RAIL_COLLAPSE_MAX_WIDTH: 767,
  useBreakpoint: () => false,
}));
vi.mock("#/stores/sidebar-store", () => ({
  useSidebarStore: (
    selector: (state: { toggleCollapsed: () => void }) => unknown,
  ) => selector({ toggleCollapsed: vi.fn() }),
}));
vi.mock("#/components/features/sidebar/sidebar-mobile-nav-context", () => ({
  useOptionalSidebarMobileNav: () => null,
}));
vi.mock("#/hooks/query/use-conversation-skills", () => ({
  useConversationSkills: () => ({ data: [], refetch: vi.fn() }),
}));

function Harness() {
  const submit = useUiCommandInterceptor(vi.fn(), {
    outputScopeId: "conversation-1",
    conversationId: "conversation-1",
    getTimelineBoundaryEventId: () => null,
  });

  return (
    <>
      <button type="button" onClick={() => submit("/skills")}>
        Submit skills
      </button>
      <SlashCommandMessages
        outputScopeId="conversation-1"
        timelineBoundaryEventId={null}
      />
    </>
  );
}

describe("/skills lifecycle integration", () => {
  afterEach(() => {
    useSlashCommandOutputStore.getState().clearAll();
    mocks.getLoadedResources.mockReset();
  });

  it("acknowledges the first submission immediately and updates that card in place", async () => {
    let resolveResources!: (value: {
      skills: never[];
      hooks: never[];
      mcps: never[];
    }) => void;
    mocks.getLoadedResources.mockReturnValue(
      new Promise((resolve) => {
        resolveResources = resolve;
      }),
    );
    render(<Harness />);

    fireEvent.click(screen.getByRole("button", { name: "Submit skills" }));

    const loading = screen.getByTestId("slash-command-skills-loading");
    const card = loading.closest("[data-status]");
    expect(card).toHaveAttribute("data-status", "loading");
    const entryTestId = card?.getAttribute("data-testid");

    act(() => resolveResources({ skills: [], hooks: [], mcps: [] }));

    await waitFor(() =>
      expect(screen.getByTestId(entryTestId!)).toHaveAttribute(
        "data-status",
        "ready",
      ),
    );
    expect(screen.getAllByTestId("slash-command-messages")).toHaveLength(1);
    expect(screen.getByTestId("slash-command-skills-list")).toBeVisible();
  });
});
