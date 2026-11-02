import type { ReactElement } from "react";
import { render } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { GitControlBarBranchButton } from "#/components/features/chat/git-control-bar-branch-button";
import { GitControlBarRepoButton } from "#/components/features/chat/git-control-bar-repo-button";
import { WorkspaceModeSelector } from "#/components/features/chat/workspace-mode-selector";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: { provider_tokens_set: {} } }),
}));

// Chip icons must take the theme's ink through `currentColor`, from the same
// colour class as the chip label. A fixed `color` (or fill/stroke) on the SVG
// overrides that inheritance and made the icons white-on-white in light themes
// (#18657).
const INK_CLASS =
  /^text-(contrast|muted|white|black|current|foreground|feedback-[a-z-]+|status-[a-z-]+|\[[^\]]+\])$/;

// The colour class an element's text and `currentColor` resolve from: the
// nearest ancestor-or-self class that sets the text colour.
function inkClass(element: Element): string | null {
  for (let node: Element | null = element; node; node = node.parentElement) {
    const ink = Array.from(node.classList).find((name) => INK_CLASS.test(name));
    if (ink) return ink;
  }
  return null;
}

function expectNoFixedColour(icon: SVGSVGElement) {
  expect(icon).not.toHaveAttribute("color");
  for (const node of [icon, ...Array.from(icon.querySelectorAll("*"))]) {
    for (const attribute of ["fill", "stroke"]) {
      const value = node.getAttribute(attribute);
      if (value !== null) expect(["none", "currentColor"]).toContain(value);
    }
  }
}

describe("git control bar chip icons", () => {
  it.each<[string, () => ReactElement, string]>([
    [
      "linked branch chip (branch and external-link icons)",
      () => (
        <GitControlBarBranchButton
          selectedBranch="main"
          selectedRepository="owner/repo"
          gitProvider="github"
        />
      ),
      "main",
    ],
    [
      "unlinked branch chip",
      () => (
        <GitControlBarBranchButton
          selectedBranch="main"
          selectedRepository={null}
          gitProvider={null}
        />
      ),
      "main",
    ],
    [
      "local workspace chip",
      () => (
        <GitControlBarRepoButton
          selectedRepository={null}
          gitProvider={null}
          workspaceName="qa-repo"
        />
      ),
      "qa-repo",
    ],
    [
      "workspace mode chip",
      () => (
        <WorkspaceModeSelector
          value="local_repo"
          options={["local_repo", "new_worktree"]}
          backendKind="local"
          onChange={() => {}}
        />
      ),
      "",
    ],
  ])(
    "%s icons take the same theme ink as the label",
    (_name, renderChip, labelText) => {
      const { container, getByText } = render(renderChip());
      const label = labelText
        ? getByText(labelText)
        : container.querySelector("button span.truncate")!;
      const labelInk = inkClass(label);
      expect(labelInk).toBe("text-contrast");

      const icons = Array.from(container.querySelectorAll("svg"));
      expect(icons.length).toBeGreaterThan(0);
      for (const icon of icons) {
        expectNoFixedColour(icon);
        expect(inkClass(icon)).toBe(labelInk);
      }
    },
  );
});
