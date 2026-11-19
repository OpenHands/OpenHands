import fs from "node:fs/promises";
import cp from "node:child_process";
import { performance } from "node:perf_hooks";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { rolldown } from "rolldown";

const root = fileURLToPath(new URL("../../", import.meta.url));
const temporaryDirectory = await fs.mkdtemp(
  join(tmpdir(), "openhands-projection-scaling-"),
);
const helperBundle = join(temporaryDirectory, "helper.mjs");
const storeBundle = join(temporaryDirectory, "store.mjs");

try {
  for (const [path, file] of [
    ["src/utils/handle-event-for-ui.ts", helperBundle],
    ["src/stores/use-event-store.ts", storeBundle],
  ]) {
    const bundle = await rolldown({
      input: root + "/" + path,
      platform: "node",
      resolve: { alias: { "#": root + "/src" } },
    });
    await bundle.write({ format: "esm", file });
    await bundle.close();
  }
  const { handleEventsForUI, handleEventForUI } = await import(
    pathToFileURL(helperBundle).href
  );
  const { useEventStore } = await import(pathToFileURL(storeBundle).href);
  let idReads = 0;
  function fixture(size) {
    return Array.from({ length: size }, (_, i) => {
      const id = String(i);
      const timestamp = new Date(Date.UTC(2026, 0, 1) + i * 1000).toISOString();
      const common = {
        get id() {
          idReads++;
          return id;
        },
        timestamp,
      };
      return i % 2 === 0
        ? {
            ...common,
            source: "agent",
            action: { kind: "TerminalAction" },
            tool_name: "terminal",
            tool_call_id: "call-" + i,
          }
        : {
            ...common,
            source: "environment",
            action_id: String(i - 1),
            observation: { kind: "TerminalObservation" },
            tool_name: "terminal",
            tool_call_id: "call-" + (i - 1),
          };
    });
  }
  // Accessors are installed after construction so object spread cannot consume them.
  function countable(events) {
    return events.map((e) => {
      const id = e.id;
      return {
        ...e,
        get id() {
          idReads++;
          return id;
        },
      };
    });
  }
  function measure(fn, count = 21) {
    for (let i = 0; i < 5; i++) fn();
    const values = [];
    for (let i = 0; i < count; i++) {
      const t = performance.now();
      fn();
      values.push(performance.now() - t);
    }
    values.sort((a, b) => a - b);
    return {
      median_ms: values[Math.floor(values.length / 2)],
      min_ms: values[0],
      max_ms: values.at(-1),
      samples: count,
    };
  }
  const rows = [];
  for (const size of [1000, 10000]) {
    const events = fixture(size);
    const output = handleEventsForUI(events);
    if (output.length !== size / 2) throw Error("wrong projection");
    const counted = countable(events);
    idReads = 0;
    handleEventsForUI(counted);
    const indexedReads = idReads;
    idReads = 0;
    counted.reduce((ui, e) => handleEventForUI(e, ui), []);
    const sequentialReads = idReads;
    const indexed = measure(() => handleEventsForUI(events));
    const sequential = measure(
      () => events.reduce((ui, e) => handleEventForUI(e, ui), []),
      7,
    );
    const backfillSamples = [];
    for (let i = 0; i < 12; i++) {
      useEventStore.getState().clearEvents();
      useEventStore
        .getState()
        .addEvents(events.slice(1), { nextPageId: "older", hasMore: true });
      const t = performance.now();
      useEventStore
        .getState()
        .addEvents([events[0]], {
          nextPageId: null,
          hasMore: false,
          prepend: true,
        });
      const elapsed = performance.now() - t;
      if (useEventStore.getState().uiEvents.length !== size / 2)
        throw Error("wrong backfill");
      if (i >= 3) backfillSamples.push(elapsed);
    }
    backfillSamples.sort((a, b) => a - b);
    rows.push({
      durable_events: size,
      projected_rows: output.length,
      indexed_id_reads: indexedReads,
      sequential_id_reads: sequentialReads,
      indexed_batch: indexed,
      sequential_fold: sequential,
      store_backfill: {
        median_ms: backfillSamples[4],
        min_ms: backfillSamples[0],
        max_ms: backfillSamples.at(-1),
        samples: backfillSamples.length,
      },
    });
  }
  const report = {
    source_commit: cp
      .execFileSync("git", ["rev-parse", "HEAD"], {
        cwd: root,
        encoding: "utf8",
      })
      .trim(),
    node: process.version,
    platform: process.platform,
    architecture: process.arch,
    kind: "Local synthetic scaling probe of actual bundled production functions; not live browser latency",
    fixture:
      "Chronological alternating unique TerminalAction/TerminalObservation records. Store backfill seeds all but the oldest action, then loads that action through addEvents with prepend=true.",
    rows,
    tenfold_growth: {
      indexed_id_reads: rows[1].indexed_id_reads / rows[0].indexed_id_reads,
      sequential_id_reads:
        rows[1].sequential_id_reads / rows[0].sequential_id_reads,
      indexed_median_time:
        rows[1].indexed_batch.median_ms / rows[0].indexed_batch.median_ms,
      sequential_median_time:
        rows[1].sequential_fold.median_ms / rows[0].sequential_fold.median_ms,
      store_backfill_median_time:
        rows[1].store_backfill.median_ms / rows[0].store_backfill.median_ms,
    },
    limitations:
      "Wall-clock timings are local and noisy. ID getter reads show the measured fixture lookup work, not a general JS operation count. Canonical reordering includes stable sorts; normal token methods are untouched. No claim of bounded DOM, memory, or payload eviction.",
  };
  console.log(JSON.stringify(report, null, 2));
} finally {
  await fs.rm(temporaryDirectory, { recursive: true, force: true });
}
