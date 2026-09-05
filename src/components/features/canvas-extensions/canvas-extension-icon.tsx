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
 * The custom icon is loaded as an `<img>` source (never inlined), so the SVG
 * runs under the browser's image sandbox and cannot execute scripts. Invalid,
 * missing, or failed icons degrade gracefully to the default without
 * affecting extension loading or the surrounding UI.
 */
export function CanvasExtensionIcon({
  extension,
  size = 18,
  className,
}: CanvasExtensionIconProps) {
  const iconPath = getCanvasExtensionIconPath(extension.manifest);
  const iconUrl = iconPath
    ? CanvasExtensionsService.buildIconUrl(extension.name, iconPath)
    : null;
  const [failed, setFailed] = React.useState(false);

  // A different extension or manifest change should re-arm the error retry.
  React.useEffect(() => {
    setFailed(false);
  }, [iconUrl]);

  if (!iconUrl || failed) {
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
          `[canvas-extensions] Failed to load custom icon for ${extension.name}: ${iconUrl}`,
        );
        setFailed(true);
      }}
    />
  );
}
