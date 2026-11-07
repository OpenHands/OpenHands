import { render, screen } from "@testing-library/react";
import { compile } from "@tailwindcss/node";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { ReactNode } from "react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { HookExecutionEventMessage } from "#/components/shared/hook-execution-event-message";
import { I18nKey } from "#/i18n/declaration";
import type {
  HookExecutionEvent,
  OpenHandsEvent,
} from "#/types/agent-server/core";

vi.mock("react-i18next", () => ({
  useTranslation: (namespace: string) => ({
    t: (key: string) =>
      namespace === "openhands" ? key : `wrong-namespace:${key}`,
  }),
}));

vi.mock("#/components/features/chat/generic-event-message", () => ({
  GenericEventMessage: ({
    title,
    details,
    success,
  }: {
    title: ReactNode;
    details: ReactNode;
    success?: "success" | "error";
  }) => (
    <section
      data-testid="generic-event-message"
      data-success={success ?? "none"}
    >
      <header>{title}</header>
      <div>{details}</div>
    </section>
  ),
}));

const makeHookEvent = (
  overrides: Partial<HookExecutionEvent> = {},
): HookExecutionEvent => ({
  id: "hook-event-1",
  timestamp: "2026-07-12T00:00:00.000Z",
  source: "hook",
  kind: "HookExecutionEvent",
  hook_event_type: "PreToolUse",
  hook_command: "npm test",
  success: true,
  blocked: false,
  exit_code: null,
  reason: null,
  tool_name: null,
  action_id: null,
  message_id: null,
  stdout: null,
  stderr: null,
  error: null,
  additional_context: null,
  hook_input: null,
  ...overrides,
});

const renderHookEvent = (overrides: Partial<HookExecutionEvent> = {}) =>
  render(<HookExecutionEventMessage event={makeHookEvent(overrides)} />);

let compiler: Awaited<ReturnType<typeof compile>>;

beforeAll(async () => {
  compiler = await compile(
    await readFile(resolve("src/tailwind.css"), "utf8"),
    { base: resolve("src"), onDependency: () => {} },
  );
});

describe("hook execution event message", () => {
  it.each([
    ["blocked", { blocked: true, success: false }, "--oh-feedback-warning"],
    ["failed", { blocked: false, success: false }, "--oh-feedback-error"],
  ] as const)(
    "builds theme feedback ink and tint for the %s badge",
    (status, event, role) => {
      renderHookEvent(event);
      const badge = screen.getByText(status);

      // Compile the component's emitted classes through the production stylesheet.
      // This checks the CSS contract without prescribing a utility class spelling.
      const css = compiler.build([...badge.classList]);
      const declarations = [
        ...css.matchAll(/\.((?:\\.|[\w-])+)\s*\{([^}]+)\}/g),
      ]
        .filter(([, selector]) =>
          badge.classList.contains(selector.replace(/\\(.)/g, "$1")),
        )
        .map(([, , body]) => body)
        .join("\n");

      const property = (name: "color" | "background-color") =>
        [
          ...declarations.matchAll(
            new RegExp(String.raw`(?:^|\n)\s*${name}:\s*([^;]+);`, "g"),
          ),
        ]
          .at(-1)?.[1]
          .trim();

      expect(property("color")).toBe(`var(${role})`);
      expect(property("background-color")).toBe(
        `color-mix(in oklab, var(${role}) 10%, transparent)`,
      );
    },
  );

  it("renders nothing for an event from another event family", () => {
    const event = {
      id: "state-event-1",
      timestamp: "2026-07-12T00:00:00.000Z",
      source: "agent",
      kind: "ConversationStateUpdateEvent",
      key: "execution_status",
      value: "idle",
    } as unknown as OpenHandsEvent;

    const { container } = render(<HookExecutionEventMessage event={event} />);

    expect(container).toBeEmptyDOMElement();
  });

  it.each([
    "PreToolUse",
    "PostToolUse",
    "UserPromptSubmit",
    "SessionStart",
    "SessionEnd",
    "Stop",
    "FutureHook",
  ])("titles %s hooks with the label and no emoji prefix", (hookType) => {
    renderHookEvent({
      hook_event_type: hookType as HookExecutionEvent["hook_event_type"],
    });

    const message = screen.getByTestId("generic-event-message");
    expect(
      message.textContent?.startsWith(
        `${I18nKey.HOOK$HOOK_LABEL}: ${hookType}`,
      ),
    ).toBe(true);
    expect(message).toHaveAttribute("data-success", "success");
  });

  it("shows no status badge for a successful hook", () => {
    renderHookEvent({ blocked: false, success: true });

    expect(screen.queryByText("ok")).not.toBeInTheDocument();
    expect(screen.queryByText("blocked")).not.toBeInTheDocument();
    expect(screen.queryByText("failed")).not.toBeInTheDocument();
  });

  it("shows all available metadata for a blocked hook", () => {
    renderHookEvent({
      hook_event_type: "PostToolUse",
      hook_command: "check-policy",
      success: false,
      blocked: true,
      exit_code: 13,
      reason: "Command denied by policy",
      tool_name: "terminal",
      stdout: "policy output",
      stderr: "policy warning",
      error: "hook failed",
      additional_context: "workspace is protected",
    });

    const message = screen.getByTestId("generic-event-message");
    expect(message).toHaveAttribute("data-success", "none");
    expect(
      message.textContent?.startsWith(
        `${I18nKey.HOOK$HOOK_LABEL}: PostToolUse`,
      ),
    ).toBe(true);
    expect(screen.getByText("(terminal)")).toBeInTheDocument();
    expect(screen.getByText("blocked")).toBeInTheDocument();
    expect(message).toHaveTextContent(`${I18nKey.HOOK$COMMAND}: check-policy`);
    expect(message).toHaveTextContent(`${I18nKey.HOOK$EXIT_CODE}: 13`);
    expect(message).toHaveTextContent(
      `${I18nKey.HOOK$BLOCKED_REASON}: Command denied by policy`,
    );
    expect(message).toHaveTextContent(
      `${I18nKey.HOOK$CONTEXT}: workspace is protected`,
    );
    expect(message).toHaveTextContent(`${I18nKey.HOOK$ERROR}: hook failed`);
    expect(message).toHaveTextContent(`${I18nKey.HOOK$OUTPUT}:policy output`);
    expect(message).toHaveTextContent(`${I18nKey.HOOK$STDERR}:policy warning`);
  });

  it("shows a failed status and truncates commands longer than 80 characters", () => {
    const longCommand = "x".repeat(100);
    renderHookEvent({
      hook_command: longCommand,
      success: false,
      reason: "not shown unless blocked",
    });

    const message = screen.getByTestId("generic-event-message");
    expect(message).toHaveAttribute("data-success", "error");
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.getByText(`${"x".repeat(77)}...`)).toBeInTheDocument();
    expect(message).not.toHaveTextContent(longCommand);
    expect(message).not.toHaveTextContent(I18nKey.HOOK$BLOCKED_REASON);
  });

  it("keeps an exactly 80-character command intact", () => {
    const command = "y".repeat(80);
    renderHookEvent({ hook_command: command });

    expect(screen.getByText(command)).toBeInTheDocument();
    expect(screen.getByTestId("generic-event-message")).not.toHaveTextContent(
      `${"y".repeat(77)}...`,
    );
  });

  it("omits a blocked-reason row when a blocked hook has no reason", () => {
    renderHookEvent({ blocked: true, reason: null });

    const message = screen.getByTestId("generic-event-message");
    expect(message).toHaveAttribute("data-success", "none");
    expect(message).not.toHaveTextContent(I18nKey.HOOK$BLOCKED_REASON);
  });

  it("omits the exit-code row when no exit code is available", () => {
    renderHookEvent({ exit_code: null });

    expect(screen.getByTestId("generic-event-message")).not.toHaveTextContent(
      I18nKey.HOOK$EXIT_CODE,
    );
  });
});
