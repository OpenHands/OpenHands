import { useCallback } from "react";
import {
  interpolateText,
  type ManifestScope,
} from "#/manifests/manifest-template";
import type {
  ExtensionManifest,
  ManifestAnalyticsEvent,
} from "#/manifests/types";
import { useTracking } from "./use-tracking";

function resolveProperties(
  properties: Record<string, string | number | boolean>,
  scope: ManifestScope,
): Record<string, string | number | boolean> {
  return Object.fromEntries(
    Object.entries(properties).map(([key, value]) => [
      key,
      typeof value === "string" ? interpolateText(value, scope) : value,
    ]),
  );
}

/**
 * Emit the analytics stages a manifest declares.
 *
 * The host knows when its own lifecycle events happen; the manifest decides
 * which of them are worth recording and under what names. Capture is
 * consent-gated by the shared telemetry client, which is what the manifest's
 * `consent: "required"` declaration asks for.
 */
export function useManifestAnalytics(manifest: ExtensionManifest) {
  const { trackManifestStage } = useTracking();

  return useCallback(
    (event: ManifestAnalyticsEvent, scope: ManifestScope = {}) => {
      manifest.analytics.stages
        .filter((stage) => stage.on === event)
        .forEach((stage) => {
          trackManifestStage({
            stageId: stage.id,
            properties: resolveProperties(stage.properties, {
              manifest,
              ...scope,
            }),
          });
        });
    },
    [manifest, trackManifestStage],
  );
}
