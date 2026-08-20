import React from "react";
import AutomationService from "#/api/automation-service/automation-service.api";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";

export const AUTOMATION_SDK_VERSION_CACHE_NAMESPACE = "automation-sdk-version";

const AUTOMATION_SDK_VERSION_CACHE_TIME_MS = 60 * 60 * 1000;
const CACHE_KEY_SEPARATOR = "\u0000";

interface AutomationSdkVersionCacheEntry {
  expiresAt: number;
  version: string | null;
}

const sdkVersionCache = new Map<string, AutomationSdkVersionCacheEntry>();
const sdkVersionRequests = new Map<string, Promise<string | null>>();

export function __resetAutomationSdkVersionCacheForTests() {
  sdkVersionCache.clear();
  sdkVersionRequests.clear();
}

function readCachedSdkVersion(cacheKey: string): string | null | undefined {
  const cached = sdkVersionCache.get(cacheKey);
  if (!cached) return undefined;

  if (cached.expiresAt <= Date.now()) {
    sdkVersionCache.delete(cacheKey);
    return undefined;
  }

  return cached.version;
}

function writeCachedSdkVersion(cacheKey: string, version: string | null) {
  sdkVersionCache.set(cacheKey, {
    expiresAt: Date.now() + AUTOMATION_SDK_VERSION_CACHE_TIME_MS,
    version,
  });
}

function getCachedOrFetchSdkVersion(cacheKey: string) {
  const cached = readCachedSdkVersion(cacheKey);
  if (cached !== undefined) return Promise.resolve(cached);

  const existingRequest = sdkVersionRequests.get(cacheKey);
  if (existingRequest) return existingRequest;

  if (typeof AutomationService.getSdkVersion !== "function") {
    writeCachedSdkVersion(cacheKey, null);
    return Promise.resolve(null);
  }

  const request = AutomationService.getSdkVersion()
    .then((version) => {
      writeCachedSdkVersion(cacheKey, version);
      return version;
    })
    .catch(() => {
      writeCachedSdkVersion(cacheKey, null);
      return null;
    })
    .finally(() => {
      sdkVersionRequests.delete(cacheKey);
    });
  sdkVersionRequests.set(cacheKey, request);
  return request;
}

export function useAutomationSdkVersion() {
  const active = useActiveBackend();
  const { backend } = active;
  const cacheKey = [
    AUTOMATION_SDK_VERSION_CACHE_NAMESPACE,
    backend.id,
    backend.kind,
    backend.host,
    active.orgId ?? "",
  ].join(CACHE_KEY_SEPARATOR);

  const [automationSdkVersion, setAutomationSdkVersion] = React.useState<
    string | null
  >(() => readCachedSdkVersion(cacheKey) ?? null);

  React.useEffect(() => {
    if (isNoBackend(backend)) {
      setAutomationSdkVersion(null);
      return undefined;
    }

    const cached = readCachedSdkVersion(cacheKey);
    if (cached !== undefined) {
      setAutomationSdkVersion(cached);
      return undefined;
    }

    let isMounted = true;
    setAutomationSdkVersion(null);
    void getCachedOrFetchSdkVersion(cacheKey).then((version) => {
      if (isMounted) {
        setAutomationSdkVersion(version);
      }
    });

    return () => {
      isMounted = false;
    };
  }, [backend, cacheKey]);

  return automationSdkVersion;
}
