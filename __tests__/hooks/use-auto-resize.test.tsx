import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useAutoResize } from "#/hooks/use-auto-resize";
import type { IMessageToSend } from "#/stores/conversation-store";

// Focused coverage for the one-shot "consume" signal that backs the fix for a
// branch prefill leaking into another conversation's composer.
describe("useAutoResize — value application / onValueApplied", () => {
  let el: HTMLDivElement;

  beforeEach(() => {
    el = document.createElement("div");
    el.contentEditable = "true";
    document.body.appendChild(el);
  });

  afterEach(() => {
    el.remove();
  });

  it("applies value.text to the element and signals onValueApplied once", async () => {
    const ref = { current: el };
    const onValueApplied = vi.fn();

    const { rerender } = renderHook(
      ({ value }: { value: IMessageToSend | undefined }) =>
        useAutoResize(ref, { value, onValueApplied }),
      { initialProps: { value: undefined as IMessageToSend | undefined } },
    );

    // No value yet -> nothing applied, no signal.
    expect(el.textContent).toBe("");
    expect(onValueApplied).not.toHaveBeenCalled();

    rerender({ value: { text: "restored draft", timestamp: 1 } });

    await waitFor(() => expect(el.textContent).toBe("restored draft"));
    expect(onValueApplied).toHaveBeenCalledTimes(1);
  });
});
