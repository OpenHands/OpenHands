import maritimeLogoUrl from "#/assets/branding/maritime-logo.png";

/**
 * Maritime's mark: a white sail on a black disc. Maritime publishes it as a
 * raster image only, so it is an <img> rather than an inline SVG component.
 */
export function MaritimeLogo({ size = 32 }: { size?: number }) {
  return (
    <img
      src={maritimeLogoUrl}
      width={size}
      height={size}
      alt=""
      aria-hidden
      draggable={false}
      className="rounded-full"
    />
  );
}
