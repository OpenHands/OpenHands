import React from "react";
import { PanelsTopLeft } from "lucide-react";
import { useCanvasExtensionIcon } from "#/hooks/query/use-canvas-extension-icon";
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
  const { data: icon } = useCanvasExtensionIcon(
    extension.name,
    Boolean(extension.manifest?.icon),
  );
  const [src, setSrc] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!icon) return undefined;
    const url = URL.createObjectURL(icon);
    setSrc(url);
    return () => {
      URL.revokeObjectURL(url);
      setSrc(null);
    };
  }, [icon]);

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
