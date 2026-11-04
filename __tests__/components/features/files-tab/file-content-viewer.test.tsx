import React from "react";
import { act, render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { compile, optimize } from "@tailwindcss/node";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FileContentViewer } from "#/components/features/files-tab/file-content-viewer";
import type { ViewMode } from "#/components/features/files-tab/view-mode";
import { useWorkspaceMutationCounter } from "#/stores/use-workspace-mutation-counter";
import { AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES } from "#/styles/agent-server-ui-style-scope";
import {
  applyColorTheme,
  COLOR_THEMES,
  DEFAULT_COLOR_THEME,
  type ColorThemeKey,
} from "#/themes/color-themes";

// Mock the *services* the file-content hook depends on — not the hook itself —
// so the real classification (text decoded, then flipped to binary on a NUL
// sniff) runs end to end through the viewer.
const useWorkspaceSessionMock = vi.fn();
vi.mock("#/hooks/query/use-workspace-session", async (importOriginal) => {
  const real =
    await importOriginal<
      typeof import("#/hooks/query/use-workspace-session")
    >();
  return {
    ...real, // keep the real joinWorkspaceUrl the hook builds its fetch URL with
    useWorkspaceSession: () => useWorkspaceSessionMock(),
  };
});

const useActiveConversationMock = vi.fn();
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

const useRuntimeIsReadyMock = vi.fn();
vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => useRuntimeIsReadyMock(),
}));

const getActiveBackendMock = vi.fn();
vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: () => getActiveBackendMock(),
}));

// The hook statically imports the cloud runtime service; stub the module so
// this local-path test never loads the real cloud/proxy machinery. The test
// uses the fetch (local) path, so downloadFile is never called or asserted.
vi.mock("#/api/runtime-service/agent-server-runtime-service", () => ({
  default: { downloadFile: vi.fn() },
}));

const fetchMock = vi.fn();

const BASE_URL =
  "https://agent.example.com/api/conversations/conv-1/workspace/";

function renderViewer(path: string, viewMode: ViewMode = "rich") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <FileContentViewer path={path} viewMode={viewMode} />
    </QueryClientProvider>,
  );
}

describe("FileContentViewer", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockReset();
    useWorkspaceSessionMock.mockReset();
    useActiveConversationMock.mockReset();
    useRuntimeIsReadyMock.mockReset();
    getActiveBackendMock.mockReset();

    useRuntimeIsReadyMock.mockReturnValue(true);
    useActiveConversationMock.mockReturnValue({
      data: {
        id: "conv-1",
        conversation_url: "https://agent.example.com/api/conversations/conv-1",
        session_api_key: "session-key",
      },
    });
    useWorkspaceSessionMock.mockReturnValue({
      data: { baseUrl: BASE_URL },
      isLoading: false,
      isError: false,
      error: null,
    });
    getActiveBackendMock.mockReturnValue({
      backend: { id: "local-1", kind: "local", host: "http://localhost:8000" },
      orgId: null,
    });
    useWorkspaceMutationCounter.setState({ count: 0 });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // The acceptance criteria require the clear message in BOTH view modes. The
  // plain-mode fallback and the rich-mode binary branch both route through
  // UnpreviewableFallback, so one parametrized spec covers both code paths.
  it.each(["rich", "plain"] as const)(
    "shows a clear unsupported-document message for an Office file (.pptx) in %s mode",
    async (viewMode) => {
      // Arrange: the workspace fileserver returns real .pptx bytes — a ZIP whose
      // header carries a NUL, so the hook classifies the file as binary.
      fetchMock.mockResolvedValue({
        ok: true,
        status: 200,
        arrayBuffer: () =>
          Promise.resolve(
            new Uint8Array([0x50, 0x4b, 0x03, 0x04, 0x00]).buffer,
          ),
      });

      // Act
      renderViewer("demo.pptx", viewMode);

      // Assert: the format-aware "no preview" message replaces the generic
      // binary fallback in both modes, so the pane is never blank.
      expect(
        await screen.findByTestId("file-content-viewer-unsupported-document"),
      ).toBeInTheDocument();
    },
  );

  it("keeps alert prose descendants readable without recoloring ordinary Markdown or links", async () => {
    // Use the real file hook, viewer, prose wrapper and Markdown components.
    const markdown = [
      "**Ordinary emphasis**",
      "",
      "> [!NOTE]",
      "> **Alert emphasis** and `alert-code`.",
      ">",
      "> ##### Alert heading",
      ">",
      "> <kbd>Alert key</kbd>",
      ">",
      "> <figure><figcaption>Alert caption</figcaption></figure>",
      ">",
      "> 1. Ordered alert item",
      ">",
      "> - Bulleted alert item",
      ">",
      "> [**Linked emphasis**](https://example.com/qa)",
      ">",
      "> > [!CAUTION]",
      "> > **Nested emphasis**",
    ].join("\n");
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: async () => new TextEncoder().encode(markdown).buffer,
    });
    act(() => applyColorTheme(DEFAULT_COLOR_THEME));
    const { container } = renderViewer("alerts.md");
    const viewer = await screen.findByTestId("file-content-viewer-markdown");
    const compiler = await compile(
      readFileSync(resolve("src/tailwind.css"), "utf8"),
      { base: resolve("src"), onDependency: () => {} },
    );
    const css = optimize(
      compiler.build(
        [...container.querySelectorAll("[class]")].flatMap((element) => [
          ...element.classList,
        ]),
      ),
    ).code;
    // Flatten build layers/supports for jsdom, retaining emitted selectors,
    // cascade order, custom properties and modern translucent color branches.
    const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].filter(
      ([, selector]) => !selector.trim().startsWith("@"),
    );
    const sheet = document.createElement("style");
    sheet.textContent = rules
      .map(([, selector, declarations]) => `${selector}{${declarations}}`)
      .join("\n");
    document.head.appendChild(sheet);
    try {
      for (const key of Object.keys(COLOR_THEMES) as ColorThemeKey[]) {
        act(() => applyColorTheme(key));
        const theme = COLOR_THEMES[key];
        const variables: Record<string, string> = {
          ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
          ...theme.scale,
          ...theme.tokens,
        };
        // jsdom retains var() in computed colors; resolve the actual inherited
        // prose variables against this shipped palette rather than guessing ink.
        const color = (element: Element, value: string): number[] => {
          const reference = /^var\((--[\w-]+)\)$/.exec(value);
          if (reference) {
            const name = reference[1];
            let resolved = variables[name];
            for (
              let owner: Element | null = element;
              !resolved && owner;
              owner = owner.parentElement
            ) {
              resolved = getComputedStyle(owner).getPropertyValue(name).trim();
            }
            expect(resolved, `${key} ${name}`).toBeTruthy();
            return color(element, resolved);
          }
          if (value === "inherit")
            return color(
              element.parentElement!,
              getComputedStyle(element.parentElement!).color,
            );
          if (value.startsWith("#"))
            return [1, 3, 5].map(
              (offset) => parseInt(value.slice(offset, offset + 2), 16) / 255,
            );
          const rgb = value.match(/^rgb\((\d+), (\d+), (\d+)\)$/);
          expect(rgb, value).not.toBeNull();
          return rgb!.slice(1).map((channel) => Number(channel) / 255);
        };
        const background = (element: Element): number[] => {
          const value = getComputedStyle(element).backgroundColor;
          if (!value || value === "rgba(0, 0, 0, 0)")
            return background(element.parentElement!);
          const mix = value.match(
            /^color-mix\(in \w+, (var\(--[\w-]+\)) ([\d.]+)%, transparent\)$/,
          );
          if (!mix) return color(element, value);
          const weight = Number(mix[2]) / 100;
          const parent = background(element.parentElement!);
          return color(element, mix[1]).map(
            (channel, index) => channel * weight + parent[index] * (1 - weight),
          );
        };
        const luminance = (channels: number[]) => {
          const linear = channels.map((channel) =>
            channel <= 0.04045
              ? channel / 12.92
              : ((channel + 0.055) / 1.055) ** 2.4,
          );
          return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
        };
        const contrast = (ink: number[], surface: number[]) => {
          const [dark, light] = [luminance(ink), luminance(surface)].sort(
            (a, b) => a - b,
          );
          return (light + 0.05) / (dark + 0.05);
        };
        const alert = viewer.querySelector(
          '[data-testid="markdown-alert-note"]',
        )!;
        for (const element of alert.querySelectorAll(
          "strong, h5, code, kbd, figcaption",
        )) {
          expect(
            contrast(
              color(element, getComputedStyle(element).color),
              background(element),
            ),
            `${key} ${element.textContent}`,
          ).toBeGreaterThanOrEqual(4.5);
        }
        for (const item of alert.querySelectorAll("li")) {
          let markerColor = getComputedStyle(item).color;
          for (const [, selector, declarations] of rules) {
            if (
              selector.includes("::marker") &&
              item.matches(
                selector
                  .replaceAll(" ::marker", " *")
                  .replaceAll("::marker", ""),
              )
            ) {
              markerColor =
                declarations.match(/(?:^|;)\s*color:\s*([^;]+);/)?.[1] ??
                markerColor;
            }
          }
          expect(
            contrast(color(item, markerColor), background(item)),
            `${key} ${item.textContent} marker`,
          ).toBeGreaterThanOrEqual(
            item.parentElement!.tagName === "OL" ? 4.5 : 3,
          );
        }
        const ordinary = screen.getByText("Ordinary emphasis");
        expect(color(ordinary, getComputedStyle(ordinary).color)).toEqual(
          color(ordinary, "var(--oh-foreground)"),
        );
        const linked = screen.getByText("Linked emphasis");
        expect(color(linked, getComputedStyle(linked).color)).toEqual(
          color(linked, "var(--oh-link)"),
        );
      }
    } finally {
      sheet.remove();
      act(() => applyColorTheme(DEFAULT_COLOR_THEME));
    }
  });
});
