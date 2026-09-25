import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { PluginLaunchPluginSection } from "#/components/features/launch/plugin-launch-plugin-section";
import { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";

function renderSection(plugin: PluginSpec) {
  return render(
    <PluginLaunchPluginSection
      plugin={plugin}
      originalIndex={0}
      isExpanded={false}
      onToggle={vi.fn()}
      getPluginDisplayName={(p) => p.source}
      onParameterChange={vi.fn()}
    />,
  );
}

describe("PluginLaunchPluginSection", () => {
  it("highlights the header button on hover with the mid-surface hover token", () => {
    renderSection({
      source: "github:owner/repo",
      parameters: { apiKey: "test-key" },
    });

    const button = screen.getByTestId("plugin-section-0");
    expect(button).toHaveClass("hover:bg-interactive-hover");
    expect(button).not.toHaveClass("hover:bg-tertiary");
  });
});
