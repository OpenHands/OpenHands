import { describe, expect, it, vi } from "vitest";
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

describe("createSeqCursor gap reporting", () => {
  it("does not report anything while there is no gap", () => {
    const onGap = vi.fn();
    const cursor = createSeqCursor(onGap);
    cursor.start(-1);
    [0, 1, 2].forEach(cursor.observe);
    expect(onGap).not.toHaveBeenCalled();
  });

  it("reports the skipped interval once when stepping over a full buffer", () => {
    const onGap = vi.fn();
    const cursor = createSeqCursor(onGap);
    cursor.start(9);
    // Buffer more than MAX_AHEAD seqs past the hole at 10.
    for (let seq = 11; seq <= 11 + 1024; seq += 1) {
      cursor.observe(seq);
    }

    // The cursor stepped over the hole at 10 and drained the contiguous
    // buffered seqs behind it.
    expect(cursor.value).toBe(1035);
    expect(onGap).toHaveBeenCalledTimes(1);
    expect(onGap).toHaveBeenCalledWith(10, 10);
  });

  it("keeps advancing and can report again for a later distinct gap", () => {
    const onGap = vi.fn();
    const cursor = createSeqCursor(onGap);
    cursor.start(-1);
    for (let seq = 1; seq <= 1 + 1024; seq += 1) {
      cursor.observe(seq);
    }
    expect(onGap).toHaveBeenCalledWith(0, 0);

    // A second hole later in the log reports separately.
    cursor.observe(2000 + 1025);
    for (let seq = 2001; seq <= 2000 + 1024; seq += 1) {
      cursor.observe(seq);
    }
    expect(onGap).toHaveBeenCalledTimes(2);
    expect(onGap).toHaveBeenLastCalledWith(1026, 2000);
  });
});
