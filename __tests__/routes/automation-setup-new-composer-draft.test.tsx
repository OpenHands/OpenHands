import { fireEvent, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { renderWithProviders } from "test-utils";
import AutomationSetupNew from "#/routes/automation-setup-new";
import {
  PENDING_AUTOMATION_SETUP_ID,
  clearAutomationFormSession,
} from "#/api/automation-form-session";
import { HOME_PROMPT_DRAFT_KEY } from "#/hooks/chat/use-draft-persistence";

// The setup form itself is not under test; the docked composer is.
vi.mock(
  "#/components/features/automations/setup/automation-setup-panel",
  () => ({
    AutomationSetupPanel: () => <div data-testid="setup-panel" />,
  }),
);

const HOME_DRAFT = "Plan my Home task";
const SETUP_PROMPT = "Summarize overnight alerts";

function renderSetupRoute() {
  return renderWithProviders(
    <MemoryRouter initialEntries={["/automations/setup"]}>
      <AutomationSetupNew />
    </MemoryRouter>,
    { navigation: { currentPath: "/automations/setup", conversationId: null } },
  );
}

function typeIntoComposer(text: string) {
  const input = screen.getByTestId("chat-input");
  input.textContent = text;
  fireEvent.input(input);
}

describe("AutomationSetupNew docked composer draft", () => {
  beforeAll(() => {
    // jsdom has no innerText, which the composer reads its text from.
    Object.defineProperty(HTMLElement.prototype, "innerText", {
      configurable: true,
      get(this: HTMLElement) {
        return this.textContent ?? "";
      },
    });
  });

  afterAll(() => {
    Reflect.deleteProperty(HTMLElement.prototype, "innerText");
  });

  beforeEach(() => {
    clearAutomationFormSession(PENDING_AUTOMATION_SETUP_ID);
    sessionStorage.clear();
    localStorage.clear();
  });

  it("opens with an empty composer even when a Home draft exists", async () => {
    sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, HOME_DRAFT);

    renderSetupRoute();

    expect((await screen.findByTestId("chat-input")).textContent).toBe("");
  });

  it("does not leave a typed setup prompt behind as the Home draft", async () => {
    const { unmount } = renderSetupRoute();
    await screen.findByTestId("chat-input");

    typeIntoComposer(SETUP_PROMPT);
    unmount();

    expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBeNull();
  });

  it("keeps an earlier Home draft while the setup composer is edited and left", async () => {
    sessionStorage.setItem(HOME_PROMPT_DRAFT_KEY, HOME_DRAFT);
    const { unmount } = renderSetupRoute();
    await screen.findByTestId("chat-input");

    typeIntoComposer(SETUP_PROMPT);
    typeIntoComposer("");
    unmount();

    expect(sessionStorage.getItem(HOME_PROMPT_DRAFT_KEY)).toBe(HOME_DRAFT);
  });
});
