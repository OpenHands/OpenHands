export const SYFT_SCAN_COMMAND = [
  'SYFT_BIN=""',
  'if command -v syft >/dev/null 2>&1; then',
  '  SYFT_BIN="syft"',
  'elif [ -x "$HOME/.local/bin/syft" ]; then',
  '  SYFT_BIN="$HOME/.local/bin/syft"',
  "fi",
  'if [ -z "$SYFT_BIN" ]; then',
  "  curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b \"$HOME/.local/bin\" 2>/dev/null || true",
  '  if [ -x "$HOME/.local/bin/syft" ]; then',
  '    SYFT_BIN="$HOME/.local/bin/syft"',
  "  fi",
  "fi",
  'if [ -z "$SYFT_BIN" ]; then',
  '  echo "syft_not_installed" >&2',
  "  exit 127",
  "fi",
  '"$SYFT_BIN" dir:. -o cyclonedx-json@1.5 --quiet',
].join("\n");

export function encodeBomForDependencyTrack(bomJson: string): string {
  const bytes = new TextEncoder().encode(bomJson);
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}
