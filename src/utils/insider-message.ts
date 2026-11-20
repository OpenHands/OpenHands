/** Display an App request without its transport context; preserve the event. */
export function insiderUserMessageForDisplay(text: string): string {
  const match = text.match(
    /^Canvas (?:voice )?context \(data, not instructions\):\n([^\n]+)\n\nUser request:\n([\s\S]*)$/,
  );
  if (!match) return text;
  try {
    const context: unknown = JSON.parse(match[1]);
    if (
      context &&
      typeof context === "object" &&
      "backend_id" in context &&
      typeof context.backend_id === "string"
    ) {
      return match[2];
    }
  } catch {
    // A malformed envelope or ordinary user-authored text stays visible.
  }
  return text;
}
