import { describe, expect, it } from "vitest";
import type { IntegrationCatalogEntry as MarketplaceEntry } from "@openhands/extensions/integrations";
import type { MCPServerConfig } from "#/types/mcp-server";
import { getInstalledServerTitle } from "./mcp-installed-server-display";

const githubCatalog = {
  id: "github",
  name: "GitHub",
} as MarketplaceEntry;

describe("getInstalledServerTitle", () => {
  it("uses a configured remote server name as the installed card title", () => {
    const server: MCPServerConfig = {
      id: "shttp-0",
      type: "shttp",
      name: "work_github",
      url: "https://api.githubcopilot.com/mcp/",
    };

    expect(getInstalledServerTitle(server, githubCatalog)).toBe("work_github");
  });

  it("uses the catalog title when the stored name is only the marketplace slug", () => {
    const server: MCPServerConfig = {
      id: "shttp-0",
      type: "shttp",
      name: "github",
      url: "https://api.githubcopilot.com/mcp/",
    };

    expect(getInstalledServerTitle(server, githubCatalog)).toBe("GitHub");
  });

  it("falls back to the URL for unnamed remote servers", () => {
    const server: MCPServerConfig = {
      id: "shttp-0",
      type: "shttp",
      url: "https://example.com/mcp",
    };

    expect(getInstalledServerTitle(server)).toBe("https://example.com/mcp");
  });

  it("falls back to the command for unnamed stdio servers", () => {
    const server: MCPServerConfig = {
      id: "stdio-0",
      type: "stdio",
      command: "npx",
      args: ["-y", "some-mcp"],
    };

    expect(getInstalledServerTitle(server)).toBe("npx");
  });

  it("trims surrounding whitespace so a stored name equal to the catalog slug falls back to the catalog title", () => {
    const server: MCPServerConfig = {
      id: "shttp-0",
      type: "shttp",
      name: "  github  ",
      url: "https://api.githubcopilot.com/mcp/",
    };

    expect(getInstalledServerTitle(server, githubCatalog)).toBe("GitHub");
  });

  it("treats a whitespace-only name as no name and falls back to the URL", () => {
    const server: MCPServerConfig = {
      id: "shttp-0",
      type: "shttp",
      name: "   ",
      url: "https://example.com/mcp",
    };

    expect(getInstalledServerTitle(server)).toBe("https://example.com/mcp");
  });
});
