import { beforeEach, expect, it } from "vitest";
import {
  sessionActivityKey,
  useSessionActivityStore,
} from "#/stores/session-activity-store";

const key = sessionActivityKey("cloud", "org", "launch-1");
beforeEach(() => useSessionActivityStore.setState({ entries: {} }));

it("retains observed steps and disclosure state across launch, task and conversation IDs", () => {
  const store = useSessionActivityStore.getState();
  store.begin(key, "initialize");
  store.observe(key, "WAITING_FOR_SANDBOX");
  store.setOpen(key, storeEntry().id, true);
  const task = sessionActivityKey("cloud", "org", "task-1");
  const conversation = sessionActivityKey("cloud", "org", "real-1");
  store.transfer(key, task);
  store.observe(task, "READY");
  store.transfer(task, conversation);
  expect(
    useSessionActivityStore.getState().entries[conversation][0],
  ).toMatchObject({
    id: storeEntry().id,
    steps: ["WAITING_FOR_SANDBOX", "READY"],
    phase: "complete",
    open: true,
  });
  expect(
    useSessionActivityStore.getState().entries[
      sessionActivityKey("cloud", "other", "real-1")
    ],
  ).toBeUndefined();
});

it("deduplicates an active resume but retains a later real resume", () => {
  const store = useSessionActivityStore.getState();
  store.begin(key, "resume");
  store.begin(key, "resume");
  expect(storeEntry().kind).toBe("resume");
  expect(useSessionActivityStore.getState().entries[key]).toHaveLength(1);
  store.observe(key, "READY");
  store.begin(key, "resume");
  expect(useSessionActivityStore.getState().entries[key]).toHaveLength(2);
  store.remove(key);
  expect(useSessionActivityStore.getState().entries[key]).toBeUndefined();
});

function storeEntry() {
  return useSessionActivityStore.getState().entries[key][0];
}
