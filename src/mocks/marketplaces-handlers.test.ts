import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_MARKETPLACE_NAME,
  defaultMarketplaceKinds,
  resetMarketplacesMockData,
} from "#/mocks/marketplaces-handlers";
import type {
  CatalogEntry,
  MarketplaceCatalog,
  MarketplaceSource,
} from "#/types/marketplaces";

const BASE = "http://127.0.0.1:8000/api/marketplaces";

afterEach(() => {
  resetMarketplacesMockData();
});

async function json<T>(method: string, path: string, body?: unknown) {
  const response = await fetch(`${BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  return {
    status: response.status,
    body: (text ? JSON.parse(text) : null) as T,
  };
}

async function listMarketplaces(): Promise<MarketplaceSource[]> {
  const { body } = await json<{ marketplaces: MarketplaceSource[] }>("GET", "");
  return body.marketplaces;
}

async function getCatalog(name: string): Promise<MarketplaceCatalog> {
  const { body } = await json<MarketplaceCatalog>("GET", `/${name}/catalog`);
  return body;
}

function entryPath(entry: CatalogEntry): string {
  return `/${entry.marketplace}/entries/${encodeURIComponent(entry.id)}`;
}

describe("Marketplaces MSW handlers", () => {
  it("seeds the default marketplace with every kind from the bundled catalog", async () => {
    const marketplaces = await listMarketplaces();
    const byName = new Map(marketplaces.map((m) => [m.name, m]));

    const defaultMarketplace = byName.get(DEFAULT_MARKETPLACE_NAME);
    expect(defaultMarketplace).toMatchObject({
      scope: "instance",
      trusted: true,
      is_default: true,
    });
    for (const kind of defaultMarketplaceKinds()) {
      expect(defaultMarketplace?.counts[kind]).toBeGreaterThan(0);
    }
    expect(byName.get("acme-internal")).toMatchObject({
      scope: "user",
      trusted: false,
      error_count: 1,
    });
  });

  it("adds a marketplace from a source, opens its catalog and removes it again", async () => {
    const added = await json<MarketplaceSource>("POST", "", {
      source: "github:obra/superpowers",
      trusted: true,
    });
    expect(added.status).toBe(201);
    expect(added.body).toMatchObject({
      name: "superpowers",
      scope: "user",
      trusted: true,
      counts: { plugin: 1, skill: 3 },
    });

    const catalog = await getCatalog("superpowers");
    expect(catalog.entries.map((e) => `${e.kind}:${e.name}`)).toContain(
      "plugin:superpowers",
    );

    const removed = await json<null>("DELETE", "/superpowers");
    expect(removed.status).toBe(204);
    expect((await listMarketplaces()).map((m) => m.name)).not.toContain(
      "superpowers",
    );
    expect((await json("GET", "/superpowers/catalog")).status).toBe(404);
  });

  it("generates a catalog for an unknown but well-formed source", async () => {
    const added = await json<MarketplaceSource>("POST", "", {
      source: "https://git.example.com/platform/agent-extensions.git",
      trusted: false,
    });
    expect(added.status).toBe(201);
    expect(added.body.name).toBe("agent-extensions");
    const total = Object.values(added.body.counts).reduce((a, b) => a + b, 0);
    expect(total).toBeGreaterThan(0);
  });

  it("rejects sources it cannot parse, duplicates and the default's removal", async () => {
    expect((await json("POST", "", { source: "not a source" })).status).toBe(
      400,
    );
    expect(
      (
        await json("POST", "", {
          source: "github:acme-corp/private-extensions",
        })
      ).status,
    ).toBe(401);
    expect(
      (
        await json("POST", "", {
          source: "github:other/repo",
          name: DEFAULT_MARKETPLACE_NAME,
        })
      ).status,
    ).toBe(409);
    expect((await json("DELETE", `/${DEFAULT_MARKETPLACE_NAME}`)).status).toBe(
      403,
    );
  });

  it("installs content that runs code only from a trusted marketplace", async () => {
    let catalog = await getCatalog("acme-internal");
    const plugin = catalog.entries.find((e) => e.name === "acme-deploy");
    const skill = catalog.entries.find((e) => e.name === "acme-oncall");
    expect(plugin?.runs_code).toBe(true);
    expect(skill?.runs_code).toBe(false);

    expect((await json("POST", `${entryPath(plugin!)}/install`)).status).toBe(
      403,
    );
    expect((await json("POST", `${entryPath(skill!)}/install`)).status).toBe(
      200,
    );

    await json("PATCH", "/acme-internal", { trusted: true });
    const installed = await json<CatalogEntry>(
      "POST",
      `${entryPath(plugin!)}/install`,
    );
    expect(installed.status).toBe(200);
    expect(installed.body.install).toMatchObject({
      state: "installed",
      enabled: true,
      policy: "track",
    });

    catalog = await getCatalog("acme-internal");
    expect(
      catalog.entries
        .filter((e) => e.install.state !== "not_installed")
        .map((e) => e.name)
        .sort(),
    ).toEqual(["acme-deploy", "acme-oncall"]);
  });

  it("flags tracked installs after a refresh and clears the flag on update", async () => {
    const catalog = await getCatalog(DEFAULT_MARKETPLACE_NAME);
    const skill = catalog.entries.find((e) => e.kind === "skill")!;
    await json("POST", `${entryPath(skill)}/install`);
    await json("PATCH", entryPath(skill), { enabled: false });

    const refreshed = await json<MarketplaceSource>(
      "POST",
      `/${DEFAULT_MARKETPLACE_NAME}/refresh`,
    );
    expect(refreshed.body.resolved_ref).not.toBe(
      catalog.marketplace.resolved_ref,
    );

    let entry = (await getCatalog(DEFAULT_MARKETPLACE_NAME)).entries.find(
      (e) => e.id === skill.id,
    )!;
    expect(entry.install).toMatchObject({
      state: "update_available",
      enabled: false,
    });

    const updated = await json<CatalogEntry>(
      "POST",
      `${entryPath(skill)}/update`,
    );
    expect(updated.body.install).toMatchObject({
      state: "installed",
      enabled: false,
      installed_ref: refreshed.body.resolved_ref,
    });

    await json("DELETE", entryPath(skill));
    entry = (await getCatalog(DEFAULT_MARKETPLACE_NAME)).entries.find(
      (e) => e.id === skill.id,
    )!;
    expect(entry.install.state).toBe("not_installed");
  });

  it("marks entries that another marketplace also ships", async () => {
    const catalog = await getCatalog("acme-internal");
    const review = catalog.entries.find(
      (e) => e.kind === "skill" && e.name === "code-review",
    );
    expect(review?.also_in).toEqual([DEFAULT_MARKETPLACE_NAME]);
  });
});
