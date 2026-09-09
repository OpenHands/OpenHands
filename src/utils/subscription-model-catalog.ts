import { deriveProfileNameFromModel } from "#/utils/derive-profile-name";
import { HOST_DETECTED_CONNECTION_PREFIX } from "#/utils/host-detected-provider-connections";

export const SUBSCRIPTION_SOURCES = [
  "chatgpt",
  "claude",
  "cursor-cli",
  "opencode",
] as const;

export type SubscriptionSource = (typeof SUBSCRIPTION_SOURCES)[number];

export interface SubscriptionModelOffer {
  source: SubscriptionSource;
  /** Native model id for that subscription (what the CLI / API expects). */
  id: string;
  label: string;
}

export interface MergedSubscriptionModel {
  /** Canonical id used to notice the same model on multiple subscriptions. */
  key: string;
  label: string;
  offers: SubscriptionModelOffer[];
}

export const CHATGPT_AUTO_PROFILE_PREFIX = "sub-chatgpt-";

const EMPTY_COUNTS: Record<SubscriptionSource, number> = {
  chatgpt: 0,
  claude: 0,
  "cursor-cli": 0,
  opencode: 0,
};

export function canonicalSubscriptionModelKey(id: string): string {
  return id
    .trim()
    .toLowerCase()
    .replace(/^(opencode|openai|anthropic)\//, "");
}

export function subscriptionModelToggleKey(
  source: SubscriptionSource,
  id: string,
): string {
  return `${source}:${id}`;
}

export function chatgptAutoProfileName(model: string): string {
  return `${CHATGPT_AUTO_PROFILE_PREFIX}${deriveProfileNameFromModel(model)}`;
}

export function isChatgptAutoProfileName(name: string): boolean {
  return name.startsWith(CHATGPT_AUTO_PROFILE_PREFIX);
}

export function mergeSubscriptionModelOffers(
  offers: SubscriptionModelOffer[],
): MergedSubscriptionModel[] {
  const groups = new Map<string, MergedSubscriptionModel>();
  for (const offer of offers) {
    const key = canonicalSubscriptionModelKey(offer.id);
    const existing = groups.get(key);
    if (!existing) {
      groups.set(key, { key, label: offer.label, offers: [offer] });
      continue;
    }
    if (
      !existing.offers.some(
        (item) => item.source === offer.source && item.id === offer.id,
      )
    ) {
      existing.offers.push(offer);
    }
  }
  return [...groups.values()].sort((a, b) => a.label.localeCompare(b.label));
}

export function countOffersBySource(
  offers: SubscriptionModelOffer[],
): Record<SubscriptionSource, number> {
  const counts = { ...EMPTY_COUNTS };
  for (const offer of offers) {
    counts[offer.source] += 1;
  }
  return counts;
}

export function parseCursorAgentModels(
  stdout: string,
): SubscriptionModelOffer[] {
  const offers: SubscriptionModelOffer[] = [];
  for (const line of stdout.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || /^available models$/i.test(trimmed)) continue;
    const match = trimmed.match(/^(\S+)\s+-\s+(.+)$/);
    if (!match) continue;
    offers.push({
      source: "cursor-cli",
      id: match[1],
      label: match[2].trim(),
    });
  }
  return offers;
}

export function parseOpenCodeModels(stdout: string): SubscriptionModelOffer[] {
  const offers: SubscriptionModelOffer[] = [];
  for (const line of stdout.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed?.includes("/")) continue;
    const label = canonicalSubscriptionModelKey(trimmed);
    offers.push({ source: "opencode", id: trimmed, label });
  }
  return offers;
}

export function claudeRegistryOffers(
  models: ReadonlyArray<{ id: string; label: string }>,
): SubscriptionModelOffer[] {
  return models
    .filter((model) => model.id && model.id !== "default")
    .map((model) => ({
      source: "claude" as const,
      id: model.id,
      label: model.label || model.id,
    }));
}

export function chatgptOffers(models: string[]): SubscriptionModelOffer[] {
  return models
    .filter((id) => id.trim().length > 0)
    .map((id) => ({
      source: "chatgpt" as const,
      id,
      label: id,
    }));
}

export function otherSourcesForOffer(
  merged: MergedSubscriptionModel[],
  offer: SubscriptionModelOffer,
): SubscriptionSource[] {
  const row = merged.find(
    (item) => item.key === canonicalSubscriptionModelKey(offer.id),
  );
  if (!row) return [];
  return row.offers
    .filter((item) => item.source !== offer.source)
    .map((item) => item.source);
}

/**
 * Host-detected subscription rows (and CLI-only stored rows) expose a catalog.
 * Stored API-key connections do not — they keep linked-profile counts.
 */
export function subscriptionSourceForConnection(connection: {
  id: string;
  provider: string;
}): SubscriptionSource | null {
  if (connection.id === `${HOST_DETECTED_CONNECTION_PREFIX}chatgpt`) {
    return "chatgpt";
  }
  if (connection.id === `${HOST_DETECTED_CONNECTION_PREFIX}anthropic`) {
    return "claude";
  }
  if (
    connection.id === `${HOST_DETECTED_CONNECTION_PREFIX}cursor-cli` ||
    connection.provider === "cursor-cli"
  ) {
    return "cursor-cli";
  }
  if (
    connection.id === `${HOST_DETECTED_CONNECTION_PREFIX}opencode` ||
    connection.provider === "opencode"
  ) {
    return "opencode";
  }
  return null;
}
