import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PromptEnhancementService from "#/api/prompt-enhancement-service";
import { ChatEnhancePromptButton } from "#/components/features/chat/chat-enhance-prompt-button";
import { PromptEnhancementPanel } from "#/components/features/chat/prompt-enhancement-panel";
import { usePromptEnhancement } from "#/hooks/chat/use-prompt-enhancement";
import type { PromptEnhancementAvailabilityState } from "#/hooks/query/use-prompt-enhancement-availability";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const AVAILABLE: PromptEnhancementAvailabilityState = {
  isAvailable: true,
  profileName: "default",
};

/** The composer's enhancement wiring around a plain contentEditable draft. */
function Composer({
  availability = AVAILABLE,
}: {
  availability?: PromptEnhancementAvailabilityState;
}) {
  const inputRef = React.useRef<HTMLDivElement>(null);
  const enhancement = usePromptEnhancement(inputRef, availability);
  return (
    <>
      <PromptEnhancementPanel enhancement={enhancement} />
      <div ref={inputRef} contentEditable data-testid="chat-input" />
      <ChatEnhancePromptButton
        availability={availability}
        hasDraftText
        isEnhancing={enhancement.state.status === "loading"}
        onEnhance={enhancement.enhance}
      />
    </>
  );
}

const draft = () => screen.getByTestId("chat-input");

function renderComposer(
  text: string,
  availability?: PromptEnhancementAvailabilityState,
) {
  render(<Composer availability={availability} />);
  draft().textContent = text;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe("prompt enhancement in the composer", () => {
  beforeEach(() => {
    // jsdom has no innerText, which the composer reads its text from.
    Object.defineProperty(HTMLElement.prototype, "innerText", {
      configurable: true,
      get(this: HTMLElement) {
        return this.textContent ?? "";
      },
    });
    // jsdom has no editing commands: the first command replaces the selected
    // draft, later ones append a line or a line break.
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: vi.fn((command: string, _ui?: boolean, text = "") => {
        const selection = window.getSelection();
        if (selection?.toString() === draft().textContent) {
          draft().textContent = "";
          selection.removeAllRanges();
        }
        draft().textContent += command === "insertLineBreak" ? "\n" : text;
        return true;
      }),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    Reflect.deleteProperty(HTMLElement.prototype, "innerText");
    Reflect.deleteProperty(document, "execCommand");
  });

  it("replaces the draft only with the accepted, edited suggestion", async () => {
    // Arrange
    const user = userEvent.setup();
    const enhanceSpy = vi
      .spyOn(PromptEnhancementService, "enhancePrompt")
      .mockResolvedValue(
        "Fix the login bug in `src/auth.ts`.\n\nKeep the tests.",
      );
    renderComposer("fix login bug src/auth.ts");

    // Act
    await user.click(screen.getByTestId("chat-enhance-prompt-button"));
    const suggestion = await screen.findByTestId("prompt-enhancement-text");
    await user.type(suggestion, " Add a test.");

    // Assert: the draft is untouched while the suggestion is under review.
    expect(enhanceSpy).toHaveBeenCalledWith(
      "default",
      "fix login bug src/auth.ts",
      expect.any(AbortSignal),
    );
    expect(draft()).toHaveTextContent("fix login bug src/auth.ts");

    await user.click(screen.getByTestId("prompt-enhancement-use"));

    expect(draft().textContent).toBe(
      "Fix the login bug in `src/auth.ts`.\n\nKeep the tests. Add a test.",
    );
    expect(
      screen.queryByTestId("prompt-enhancement-panel"),
    ).not.toBeInTheDocument();
  });

  it("keeps the original draft when the user keeps it", async () => {
    // Arrange
    const user = userEvent.setup();
    vi.spyOn(PromptEnhancementService, "enhancePrompt").mockResolvedValue(
      "A different prompt.",
    );
    renderComposer("my draft");

    // Act
    await user.click(screen.getByTestId("chat-enhance-prompt-button"));
    await screen.findByTestId("prompt-enhancement-text");
    await user.click(screen.getByTestId("prompt-enhancement-keep-original"));

    // Assert
    expect(draft()).toHaveTextContent("my draft");
    expect(document.execCommand).not.toHaveBeenCalled();
    expect(
      screen.queryByTestId("prompt-enhancement-panel"),
    ).not.toBeInTheDocument();
  });

  it("drops a late suggestion when the draft changed during the request", async () => {
    // Arrange
    const user = userEvent.setup();
    const response = deferred<string>();
    vi.spyOn(PromptEnhancementService, "enhancePrompt").mockReturnValue(
      response.promise,
    );
    renderComposer("first draft");

    // Act
    await user.click(screen.getByTestId("chat-enhance-prompt-button"));
    draft().textContent = "first draft, edited";
    response.resolve("Enhanced first draft.");

    // Assert
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "CHAT_INTERFACE$ENHANCE_PROMPT_ERROR_DRAFT_CHANGED",
    );
    expect(draft()).toHaveTextContent("first draft, edited");
    expect(
      screen.queryByTestId("prompt-enhancement-text"),
    ).not.toBeInTheDocument();
  });

  it("shows the server error and leaves the draft unchanged", async () => {
    // Arrange
    const user = userEvent.setup();
    vi.spyOn(PromptEnhancementService, "enhancePrompt").mockRejectedValue(
      Object.assign(new Error("Gateway Timeout"), {
        name: "HttpError",
        status: 504,
        response: { code: "enhancement_timeout", message: "Timed out." },
      }),
    );
    renderComposer("my draft");

    // Act
    await user.click(screen.getByTestId("chat-enhance-prompt-button"));

    // Assert
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "CHAT_INTERFACE$ENHANCE_PROMPT_ERROR_TIMEOUT",
    );
    expect(draft()).toHaveTextContent("my draft");
  });

  it("aborts the request on cancel and ignores its response", async () => {
    // Arrange
    const user = userEvent.setup();
    const response = deferred<string>();
    const enhanceSpy = vi
      .spyOn(PromptEnhancementService, "enhancePrompt")
      .mockReturnValue(response.promise);
    renderComposer("my draft");

    // Act
    await user.click(screen.getByTestId("chat-enhance-prompt-button"));
    await user.click(screen.getByTestId("prompt-enhancement-cancel"));
    response.resolve("Too late.");

    // Assert
    const signal = enhanceSpy.mock.calls[0][2];
    expect(signal.aborted).toBe(true);
    await waitFor(() =>
      expect(
        screen.queryByTestId("prompt-enhancement-panel"),
      ).not.toBeInTheDocument(),
    );
    expect(draft()).toHaveTextContent("my draft");
  });

  it("does not request an enhancement when it is unavailable", async () => {
    // Arrange
    const user = userEvent.setup();
    const enhanceSpy = vi.spyOn(PromptEnhancementService, "enhancePrompt");
    renderComposer("my draft", {
      isAvailable: false,
      reason: "unsupported_backend",
    });
    const button = screen.getByTestId("chat-enhance-prompt-button");

    // Act
    await user.click(button);

    // Assert
    expect(button).toHaveAttribute("aria-disabled", "true");
    expect(button).toHaveAttribute(
      "data-unavailable-reason",
      "unsupported_backend",
    );
    expect(enhanceSpy).not.toHaveBeenCalled();
  });
});
