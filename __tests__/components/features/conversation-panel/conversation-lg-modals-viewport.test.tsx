import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import { parse } from "postcss";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import SkillsService from "#/api/skills-service";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { HooksModal } from "#/components/features/conversation-panel/hooks-modal";
import { SkillsModal } from "#/components/features/conversation-panel/skills-modal";
import { SystemMessageModal } from "#/components/features/conversation-panel/system-message-modal";
import { MODAL_WIDTH_CLASS } from "#/components/shared/modals/modal-body";
import { adaptSystemMessage } from "#/utils/system-message-adapter";
import { EventState } from "#/stores/use-event-store";

const LONG_SYSTEM_PROMPT = "v1 prompt\n".repeat(80);

const systemPromptEvents: EventState["events"] = [
  {
    id: "v1-id",
    timestamp: "2025-12-30T12:00:00Z",
    source: "agent",
    system_prompt: {
      type: "text",
      text: LONG_SYSTEM_PROMPT,
    },
    tools: [
      {
        type: "function",
        function: {
          name: "bash",
          description: "Execute bash",
          parameters: {},
        },
      },
    ],
  },
];

const systemMessage = adaptSystemMessage(systemPromptEvents);

const DIALOGS = [
  {
    name: "SkillsModal",
    testId: "skills-modal",
    closeTestId: "close-skills-modal",
    renderDialog: (onClose: () => void) => <SkillsModal onClose={onClose} />,
  },
  {
    name: "HooksModal",
    testId: "hooks-modal",
    closeTestId: "close-hooks-modal",
    renderDialog: (onClose: () => void) => <HooksModal onClose={onClose} />,
  },
  {
    name: "SystemMessageModal",
    testId: "system-message-modal",
    closeTestId: "close-system-message-modal",
    renderDialog: (onClose: () => void) => (
      <SystemMessageModal
        isOpen
        onClose={onClose}
        systemMessage={systemMessage}
      />
    ),
  },
] as const;

describe("conversation lg dialogs viewport fit", () => {
  beforeEach(() => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([]);
    vi.spyOn(AgentServerConversationService, "getHooks").mockResolvedValue({
      hooks: [],
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("keeps the shared lg token at 640px so a 720px viewport still uses the desktop width", () => {
    // w-160 is 40rem/640px; 90vw at 720px is 648px, so min(640px, 90vw) remains 640px.
    expect(MODAL_WIDTH_CLASS.lg).toBe("w-160");
  });

  it.each(DIALOGS)(
    "$name caps the 640px body at 90vw so 320px and 390px viewports cannot overflow",
    async ({ testId, renderDialog }) => {
      renderWithProviders(renderDialog(vi.fn()));

      const body = await screen.findByTestId(testId);
      expect(body).toHaveClass(MODAL_WIDTH_CLASS.lg);

      const scrollRegion = body.querySelector(".overflow-auto");
      if (!scrollRegion) throw new Error("Expected the dialog scroll region");
      const compiler = await compile(
        await readFile(resolve("src/tailwind.css"), "utf8"),
        { base: resolve("src"), onDependency: () => {} },
      );
      const css = parse(
        compiler.build([...body.classList, ...scrollRegion.classList]),
      );
      for (const [element, property, expected] of [
        [body, "max-width", "90vw"],
        [scrollRegion, "height", "60vh"],
      ] as const) {
        const values: string[] = [];
        css.walkDecls(property, (declaration) => {
          const rule = declaration.parent;
          if (rule?.type === "rule" && element.matches(rule.selector))
            values.push(declaration.value);
        });
        expect(values).toContain(expected);
      }
    },
  );

  it.each(DIALOGS)(
    "$name close control is visible, focusable, clickable, and Escape dismisses",
    async ({ closeTestId, renderDialog }) => {
      const user = userEvent.setup();
      const onClose = vi.fn();
      renderWithProviders(renderDialog(onClose));

      const closeButton = await screen.findByTestId(closeTestId);
      expect(closeButton).toBeVisible();
      expect(closeButton).toBeEnabled();
      closeButton.focus();
      expect(closeButton).toHaveFocus();

      await user.click(closeButton);
      expect(onClose).toHaveBeenCalledTimes(1);

      onClose.mockClear();
      await user.keyboard("{Escape}");
      expect(onClose).toHaveBeenCalledTimes(1);
    },
  );
});
