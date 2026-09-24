import React from "react";
import { PanelsTopLeft } from "lucide-react";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import type { InstalledCanvasExtensionInfo } from "#/types/canvas-extension";
import { getCanvasExtensionIconPath } from "#/utils/canvas-extension-icon";

interface CanvasExtensionIconProps {
  extension: InstalledCanvasExtensionInfo;
  /** Width and height in px; both the custom `<img>` and the default icon. */
  size?: number;
  className?: string;
}

/**
 * Icon for an installed Canvas Extension. Renders the extension's
 * manifest-declared SVG icon when one is present and loadable, and falls back
 * to the default extension icon otherwise.
 *
 * The custom icon is fetched through the active backend's typed client
 * (so it authenticates with the session API key and targets the selected
 * backend host) and rendered as an `<img>` object-URL source (never inlined,
 * so the SVG runs under the browser's image sandbox and cannot execute scripts.
 * Invalid, missing, or failed icons degrade gracefully to the default
 * without affecting extension loading or the surrounding UI.
 */
export function CanvasExtensionIcon({
  extension,
  size = 18,
  className,
}: CanvasExtensionIconProps) {
  const iconPath = getCanvasExtensionIconPath(extension.manifest);
  const [iconUrl, setIconUrl] = React.useState<string | null>(null);
  const [failed, setFailed] = React.useState(false);

  React.useEffect(() => {
    if (!iconPath) {
      setFailed(false);
      return;
    }

    let cancelled = false;
    let objectUrl: string | null = null;

    setFailed(false);
    setIconUrl(null);

    (async () => {
      try {
        const blob = await CanvasExtensionsService.fetchIcon(
          extension.name,
          iconPath,
        );
        if (cancelled) return;
        if (!blob) throw new Error("Icon asset unavailable");
        objectUrl = URL.createObjectURL(blob);
        if (cancelled) {
          URL.revokeObjectURL(objectUrl);
          return;
        }
        setIconUrl(objectUrl);
      } catch (error) {
        if (cancelled) return;
        console.warn(
          `[canvas-extensions] Failed to load custom icon for ${extension.name}: ${iconPath}`,
          error,
        );
        setFailed(true);
      }
    })();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [extension.name, iconPath]);

  if (!iconPath || failed) {
    return (
      <PanelsTopLeft
        width={size}
        height={size}
        className={className}
        aria-hidden
        data-testid="canvas-extension-default-icon"
      />
    );
  }

  if (!iconUrl) return null;

  return (
    <img
      src={iconUrl}
      width={size}
      height={size}
      alt=""
      aria-hidden
      className={className}
      data-testid="canvas-extension-icon"
      onError={() => {
        console.warn(
          `[canvas-extensions] Failed to load custom icon for ${extension.name}: ${iconPath}`,
        );
        setFailed(true);
      }}
    />
  );
}
