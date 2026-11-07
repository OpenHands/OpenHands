(() => {
  const panel = document.querySelector('[data-testid="usage-panel"]');
  const fills = [...(panel?.querySelectorAll("[style]") ?? [])].filter(
    (e) =>
      e.classList.contains("transition-all") &&
      e.parentElement.classList.contains("h-1.5") &&
      e.parentElement.classList.contains("bg-tertiary") &&
      e.parentElement.parentElement.querySelector("span")?.textContent.trim().startsWith("$"),
  );
  if (fills.length !== 1) return { meterCount: fills.length };
  const fill = fills[0],
    track = fill.parentElement;
  const text = track.parentElement.querySelector("span");
  const surface = track.closest(".bg-surface-raised");
  const fs = getComputedStyle(fill),
    ts = getComputedStyle(track);
  const luminance = (color) => {
    if (!/^rgba?\(/.test(color)) return null;
    const n = color.match(/[\d.]+/g).map(Number);
    if (n.length > 3 && n[3] !== 1) return null;
    return n
      .slice(0, 3)
      .map((v) => v / 255)
      .map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4))
      .reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0);
  };
  const contrast = (a, b) => {
    const x = luminance(a),
      y = luminance(b);
    return x === null || y === null
      ? null
      : (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
  };
  return {
    meterCount: fills.length,
    budgetText: text?.textContent.trim(),
    inlineWidth: fill.style.width,
    fillWidth: fill.getBoundingClientRect().width,
    trackWidth: track.getBoundingClientRect().width,
    fillColor: fs.backgroundColor,
    trackColor: ts.backgroundColor,
    fillTrackContrast: contrast(fs.backgroundColor, ts.backgroundColor),
    foregroundRole: fs.getPropertyValue("--oh-foreground").trim(),
    errorRole: fs.getPropertyValue("--oh-status-error").trim(),
    textColor: text ? getComputedStyle(text).color : null,
    surfaceColor: surface ? getComputedStyle(surface).backgroundColor : null,
    textSurfaceContrast:
      text && surface
        ? contrast(
            getComputedStyle(text).color,
            getComputedStyle(surface).backgroundColor,
          )
        : null,
  };
})()
