import { createElement } from "react";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import MCPRoute from "#/routes/mcp";
import LlmSettingsRoute from "#/routes/llm-settings";
import SecretsSettingsRoute from "#/routes/secrets-settings";
import SkillsSettingsRoute from "#/routes/skills-settings";

const legacyRoutes = [
  ["/mcp", MCPRoute],
  ["/skills", SkillsSettingsRoute],
  ["/settings/llm", LlmSettingsRoute],
  ["/settings/secrets", SecretsSettingsRoute],
] as const;

describe("legacy agent configuration routes", () => {
  it.each(legacyRoutes)("redirects %s to Agents", async (path, Component) => {
    render(
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path={path} element={createElement(Component)} />
          <Route path="/agents" element={<p>Agents</p>} />
        </Routes>
      </MemoryRouter>,
    );
    expect(await screen.findByText("Agents")).toBeInTheDocument();
  });
});
