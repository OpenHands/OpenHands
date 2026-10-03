import React from "react";
import { PanelsTopLeft } from "lucide-react";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import type { InstalledCanvasExtensionInfo } from "#/types/canvas-extension";

interface CanvasExtensionIconProps {
  extension: InstalledCanvasExtensionInfo;
  size: number;
  className?: string;
}

// Rendered via <img> (never inlined) so the SVG cannot run scripts.
export function CanvasExtensionIcon({
  extension,
  size,
  className,
}: CanvasExtensionIconProps) {
  const icon = extension.manifest?.icon;
  const [src, setSrc] = React.useState<string | null>(null);

  React.useEffect(() => {
    setSrc(null);
    if (!icon) return undefined;
    let cancelled = false;
    let objectUrl: string | null = null;
    CanvasExtensionsService.fetchIcon(extension.name)
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setSrc(objectUrl);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [extension.name, icon]);

  if (!src) {
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
      src={src}
      width={size}
      height={size}
      alt=""
      aria-hidden
      className={className}
      data-testid="canvas-extension-icon"
      onError={() => setSrc(null)}
    />
  );
}
