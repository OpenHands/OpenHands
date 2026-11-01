import { describe, expect, it } from "vitest";
import { createSeqCursor } from "#/utils/session-seq-cursor";

describe("createSeqCursor", () => {
  it("is null until a connection starts it", () => {
    const cursor = createSeqCursor();
    cursor.observe(5);
    expect(cursor.value).toBeNull();
  });

  it("advances through contiguous seqs", () => {
    const cursor = createSeqCursor();
    cursor.start(-1);
    [0, 1, 2].forEach(cursor.observe);
    expect(cursor.value).toBe(2);
  });

  it("holds below a gap and closes it once the missing seq arrives", () => {
    const cursor = createSeqCursor();
    cursor.start(3);
    cursor.observe(5);
    cursor.observe(6);
    expect(cursor.value).toBe(3);

    cursor.observe(4);
    expect(cursor.value).toBe(6);
  });

  it("ignores replays at or below the cursor", () => {
    const cursor = createSeqCursor();
    cursor.start(10);
    cursor.observe(10);
    cursor.observe(2);
    expect(cursor.value).toBe(10);
  });

  it("forgets seqs past a gap on restart, since they are replayed", () => {
    const cursor = createSeqCursor();
    cursor.start(0);
    cursor.observe(2);
    cursor.start(cursor.value ?? -1);
    cursor.observe(1);
    expect(cursor.value).toBe(1);
  });

  it("clears back to no position", () => {
    const cursor = createSeqCursor();
    cursor.start(7);
    cursor.clear();
    expect(cursor.value).toBeNull();
  });
});
