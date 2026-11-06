import { AgentServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "./agent-server-client-options";
import { getActiveBackend } from "./backend-registry/active-store";
import type {
  AddMarketplaceRequest,
  CatalogEntry,
  MarketplaceCatalog,
  MarketplaceSource,
  UpdateMarketplaceRequest,
} from "#/types/marketplaces";

/**
 * Marketplace sources and their catalogs on the local agent-server.
 *
 * Prototype: the endpoints under `/api/marketplaces` exist only in the mock
 * API (`npm run dev:mock`). The typed client does not cover them yet, so the
 * generic `AgentServerClient` methods are used, as the Canvas Apps service
 * does.
 */
export const MARKETPLACES_BASE_PATH = "/api/marketplaces";

function isCloudBackend(): boolean {
  return getActiveBackend().backend.kind === "cloud";
}

function getClient(): AgentServerClient {
  return new AgentServerClient(getAgentServerClientOptions());
}

function encode(value: string): string {
  return encodeURIComponent(value);
}

class MarketplacesService {
  static async list(): Promise<MarketplaceSource[]> {
    if (isCloudBackend()) return [];
    const response = await getClient().get<{
      marketplaces: MarketplaceSource[];
    }>(MARKETPLACES_BASE_PATH);
    return response.marketplaces ?? [];
  }

  static async add(request: AddMarketplaceRequest): Promise<MarketplaceSource> {
    return getClient().post<MarketplaceSource>(MARKETPLACES_BASE_PATH, request);
  }

  static async update(
    name: string,
    request: UpdateMarketplaceRequest,
  ): Promise<MarketplaceSource> {
    return getClient().patch<MarketplaceSource>(
      `${MARKETPLACES_BASE_PATH}/${encode(name)}`,
      request,
    );
  }

  static async remove(name: string): Promise<void> {
    await getClient().delete<unknown>(
      `${MARKETPLACES_BASE_PATH}/${encode(name)}`,
    );
  }

  static async refresh(name: string): Promise<MarketplaceSource> {
    return getClient().post<MarketplaceSource>(
      `${MARKETPLACES_BASE_PATH}/${encode(name)}/refresh`,
      {},
    );
  }

  static async getCatalog(name: string): Promise<MarketplaceCatalog> {
    return getClient().get<MarketplaceCatalog>(
      `${MARKETPLACES_BASE_PATH}/${encode(name)}/catalog`,
    );
  }

  static async installEntry(
    marketplace: string,
    entryId: string,
  ): Promise<CatalogEntry> {
    return getClient().post<CatalogEntry>(
      `${MARKETPLACES_BASE_PATH}/${encode(marketplace)}/entries/${encode(entryId)}/install`,
      {},
    );
  }

  static async updateEntry(
    marketplace: string,
    entryId: string,
  ): Promise<CatalogEntry> {
    return getClient().post<CatalogEntry>(
      `${MARKETPLACES_BASE_PATH}/${encode(marketplace)}/entries/${encode(entryId)}/update`,
      {},
    );
  }

  static async setEntryEnabled(
    marketplace: string,
    entryId: string,
    enabled: boolean,
  ): Promise<CatalogEntry> {
    return getClient().patch<CatalogEntry>(
      `${MARKETPLACES_BASE_PATH}/${encode(marketplace)}/entries/${encode(entryId)}`,
      { enabled },
    );
  }

  static async uninstallEntry(
    marketplace: string,
    entryId: string,
  ): Promise<CatalogEntry> {
    return getClient().delete<CatalogEntry>(
      `${MARKETPLACES_BASE_PATH}/${encode(marketplace)}/entries/${encode(entryId)}`,
    );
  }
}

export default MarketplacesService;
