/** The agent tool and persisted drafts share the comma-separated form value. */
export function parseAutomationEventKeys(value: string): string[] {
  return [
    ...new Set(
      value
        .split(",")
        .map((key) => key.trim())
        .filter(Boolean),
    ),
  ];
}

export function buildAutomationEventTrigger(
  source: string,
  eventKey: string,
  filter: string,
  existing?: Record<string, unknown>,
) {
  const keys = parseAutomationEventKeys(eventKey);
  const trigger: Record<string, unknown> = {
    ...existing,
    type: "event",
    source,
    on: keys.length === 1 ? keys[0] : keys,
  };
  if (filter.trim()) trigger.filter = filter;
  return trigger;
}
