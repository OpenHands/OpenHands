// Paste into the console of a Canvas dev server (`npm run dev:mock`) on a
// conversation page with the Files tab open. Replays 200 TerminalObservation
// events into the event store 10ms apart, then counts workspace listings.
(async () => {
  performance.setResourceTimingBufferSize(100000);
  performance.clearResourceTimings();
  const { useEventStore } = await import("/src/stores/use-event-store.ts");
  const run = Math.random().toString(36).slice(2, 8);
  for (let i = 1; i <= 200; i += 1) {
    useEventStore.getState().addEvent({
      id: `qa-${run}-${i}`,
      timestamp: new Date(Date.UTC(2026, 10, 27, 14) + i * 1000).toISOString(),
      source: "environment",
      tool_name: "terminal",
      tool_call_id: `qa-tc-${run}-${i}`,
      action_id: `qa-act-${run}-${i}`,
      observation: {
        kind: "TerminalObservation",
        command: "ls",
        content: [{ type: "text", text: "" }],
        exit_code: 0,
        is_error: false,
        timeout: false,
      },
    });
    await new Promise((r) => setTimeout(r, 10));
  }
  await new Promise((r) => setTimeout(r, 3000));
  const listings = performance
    .getEntriesByType("resource")
    .filter((e) => e.name.includes("/api/bash/execute_bash_command")).length;
  console.log({ events: 200, listings });
})();
