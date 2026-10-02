import { render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MonoComponent } from "#/components/features/chat/mono-component";
import EventLogger from "#/utils/event-logger";

const getMonoElement = (container: HTMLElement): HTMLElement => {
  const element = container.querySelector("strong");
  if (!(element instanceof HTMLElement)) {
    throw new Error("Expected MonoComponent to render a strong element");
  }
  return element;
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("MonoComponent", () => {
  it("decodes HTML entities as inert monospace text", () => {
    const { container } = render(
      <MonoComponent>
        Run &amp;amp; &#38;#60;script&#38;#62;alert(1)&#38;#60;/script&#38;#62;
      </MonoComponent>,
    );

    const mono = getMonoElement(container);
    expect(mono).toHaveClass("font-mono");
    expect(mono.textContent).toBe("Run & <script>alert(1)</script>");
    expect(container.querySelector("script")).not.toBeInTheDocument();
  });

  it("decodes string entries in mixed children while preserving other nodes", () => {
    const children = ["A&amp;B", " / ", <em key="nested">nested</em>, 7];
    const { container } = render(<MonoComponent>{children}</MonoComponent>);

    const mono = getMonoElement(container);
    expect(mono).toHaveClass("font-mono");
    expect(mono.textContent).toBe("A&B / nested7");
    expect(mono.querySelector("em")).toHaveTextContent("nested");
  });

  it("preserves a non-string child inside the monospace wrapper", () => {
    const { container, getByTestId } = render(
      <MonoComponent>
        <span data-testid="nested-child">unchanged</span>
      </MonoComponent>,
    );

    const mono = getMonoElement(container);
    const child = getByTestId("nested-child");
    expect(mono).toHaveClass("font-mono");
    expect(mono).toContainElement(child);
    expect(child).toHaveTextContent("unchanged");
  });

  it("logs decoding failures and preserves the original string", () => {
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation(((
      tagName: string,
      options?: ElementCreationOptions,
    ) => {
      if (tagName === "textarea") {
        throw new Error("decode failed");
      }
      return originalCreateElement(tagName, options);
    }) as typeof document.createElement);
    const error = vi.spyOn(EventLogger, "error").mockImplementation(() => {});

    const { container } = render(<MonoComponent>{"&amp;"}</MonoComponent>);

    const mono = getMonoElement(container);
    expect(mono).toHaveClass("font-mono");
    expect(mono.textContent).toBe("&amp;");
    expect(error).toHaveBeenCalledOnce();
    expect(error).toHaveBeenCalledWith("Error: decode failed");
  });
});
