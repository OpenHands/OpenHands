import { I18nKey } from "#/i18n/declaration";

export const DIGITALOCEAN_PROVIDER_ID = "digitalocean" as const;

export type ThirdPartyBackendProviderId = typeof DIGITALOCEAN_PROVIDER_ID;

export interface ThirdPartyBackendProvider {
  id: ThirdPartyBackendProviderId;
  /** Brand name — not localized. */
  name: string;
  descriptionKey: I18nKey;
}

export const THIRD_PARTY_BACKEND_PROVIDERS: readonly ThirdPartyBackendProvider[] =
  [
    {
      id: DIGITALOCEAN_PROVIDER_ID,
      name: "DigitalOcean",
      descriptionKey: I18nKey.BACKEND$PROVIDER_DIGITALOCEAN_DESCRIPTION,
    },
  ];

/** OpenHands 1-Click App. DigitalOcean prompts signup when the user is logged out. */
export const DIGITALOCEAN_MARKETPLACE_URL =
  "https://marketplace.digitalocean.com/apps/openhands";

/**
 * Static droplets for the DigitalOcean connect mock. These are display-only
 * fixtures — the panel never calls DigitalOcean or adds a backend.
 */
export interface MockDigitalOceanDroplet {
  id: string;
  name: string;
  region: string;
  ipv4: string;
  status: "active" | "off";
}

export const MOCK_DIGITALOCEAN_DROPLETS: readonly MockDigitalOceanDroplet[] = [
  {
    id: "do-droplet-nyc3",
    name: "openhands-nyc3",
    region: "NYC3",
    ipv4: "167.172.14.22",
    status: "active",
  },
  {
    id: "do-droplet-sfo3",
    name: "openhands-sfo3",
    region: "SFO3",
    ipv4: "143.198.61.9",
    status: "active",
  },
];
