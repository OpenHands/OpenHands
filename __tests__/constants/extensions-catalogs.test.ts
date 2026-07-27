import { describe, expect, it } from "vitest";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import { INTEGRATION_CATALOG } from "@openhands/extensions/integrations";
import {
  getDefaultMcpTransport,
  getInstallableMcpConnectionOption,
  getMcpMarketplaceCatalog,
} from "#/utils/mcp-marketplace-utils";

describe("OpenHands extensions catalogs", () => {
  it("hydrates the MCP marketplace from @openhands/extensions", () => {
    expect(INTEGRATION_CATALOG.length).toBeGreaterThan(0);

    const github = INTEGRATION_CATALOG.find((entry) => entry.id === "github");
    expect(getDefaultMcpTransport(github!)?.kind).toBe("shttp");
    expect(github?.logoUrl).toBe("https://cdn.simpleicons.org/github/FFFFFF");
  });

  it("patches Slack to the maintained docs and npm package", () => {
    const slack = INTEGRATION_CATALOG.find((entry) => entry.id === "slack");
    expect(slack?.docsUrl).toBe(
      "https://github.com/zencoderai/slack-mcp-server",
    );
    const apiOption = slack?.connectionOptions.find(
      (option) => option.id === "api" && option.transport?.kind === "stdio",
    );
    expect(apiOption?.transport?.kind).toBe("stdio");
    if (apiOption?.transport?.kind !== "stdio") {
      throw new Error("Slack API option should be stdio");
    }
    expect(apiOption.transport.args).toContain("@zencoderai/slack-mcp-server");
    expect(apiOption.transport.args).not.toContain(
      "@modelcontextprotocol/server-slack",
    );
  });

  it("loads Linear streamable HTTP /mcp endpoint with bearer auth", () => {
    const catalog = getMcpMarketplaceCatalog(INTEGRATION_CATALOG);
    const linear = catalog.find((entry) => entry.id === "linear")!;

    const mcpOption = getInstallableMcpConnectionOption(linear)!;

    expect(mcpOption.transport).toEqual({
      kind: "shttp",
      url: "https://mcp.linear.app/mcp",
      apiKeyOptional: true,
    });
    expect(linear.docsUrl).toBe("https://linear.app/docs/mcp");
    expect(mcpOption.auth.strategy).toBe("bearer");
    expect(
      linear.connectionOptions.some((option) => option.transport?.kind === "sse"),
    ).toBe(false);
  });

  it("drops deprecated MCP entries that no longer have maintained replacements", () => {
    const catalogIds = new Set(
      getMcpMarketplaceCatalog(INTEGRATION_CATALOG).map((entry) => entry.id),
    );

    expect(catalogIds.has("gitlab")).toBe(false);
    expect(catalogIds.has("google-maps")).toBe(false);
    expect(catalogIds.has("postgres")).toBe(false);
    expect(catalogIds.has("puppeteer")).toBe(false);
    expect(catalogIds.has("sqlite")).toBe(false);
  });

  it("loads recommended automations from @openhands/extensions", () => {
    expect(AUTOMATION_CATALOG.length).toBeGreaterThan(0);

    const knownMcpIds = new Set(INTEGRATION_CATALOG.map((entry) => entry.id));
    for (const automation of AUTOMATION_CATALOG) {
      expect(automation.requiredIntegrationIds.length).toBeGreaterThan(0);
      expect(
        automation.requiredIntegrationIds.every((id) => knownMcpIds.has(id)),
      ).toBe(true);
    }
  });
});
