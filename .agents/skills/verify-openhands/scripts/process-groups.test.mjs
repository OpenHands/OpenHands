import assert from "node:assert/strict";
import {
  mkdtempSync,
  mkdirSync,
  writeFileSync,
  rmSync,
  readdirSync,
  readFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "vitest";
import { groupAlive, privatePurgeAllowed } from "./lib/process-groups.mjs";

function fixture(rows) {
  const root = mkdtempSync(join(tmpdir(), "oh-process-groups-"));
  for (const [pid, group, state] of rows) {
    mkdirSync(join(root, String(pid)));
    // A legal comm may contain whitespace and closing parentheses.
    writeFileSync(
      join(root, String(pid), "stat"),
      `${pid} (worker ) helper) ${state} 1 ${group} 0 0`,
    );
  }
  return {
    root,
    options: { platform: "linux", procRoot: root, signal: () => {} },
    clean: () => rmSync(root, { recursive: true, force: true }),
  };
}

test("all zombie/dead group is stopped, but mixed live members keep it alive", () => {
  const f = fixture([
    [101, 77, "Z"],
    [102, 77, "X"],
    [103, 88, "S"],
  ]);
  try {
    assert.equal(groupAlive(77, f.options), false);
    assert.equal(groupAlive(88, f.options), true);
    writeFileSync(join(f.root, "102", "stat"), "102 (worker) S 1 77 0");
    assert.equal(groupAlive(77, f.options), true);
  } finally {
    f.clean();
  }
});

test("permission, unreadable and unobserved groups fail closed", () => {
  const f = fixture([[101, 77, "Z"]]);
  try {
    assert.equal(
      groupAlive(77, {
        ...f.options,
        signal: () => {
          throw Object.assign(new Error(), { code: "EPERM" });
        },
      }),
      true,
    );
    assert.equal(
      groupAlive(77, {
        ...f.options,
        readStat: () => {
          throw Object.assign(new Error(), { code: "ENOENT" });
        },
      }),
      true,
    );
    assert.equal(
      groupAlive(77, {
        ...f.options,
        readDirectory: () => {
          throw new Error("denied");
        },
      }),
      true,
    );
    assert.equal(groupAlive(99, f.options), true);
  } finally {
    f.clean();
  }
});

test.each(["appearing", "exiting"])(
  "unrelated %s processes do not keep a zombie-only group alive",
  (churn) => {
    const f = fixture([
      [101, 77, "Z"],
      [201, 88, "S"],
    ]);
    let scans = 0;
    try {
      assert.equal(
        groupAlive(77, {
          ...f.options,
          readDirectory: (root) => {
            scans++;
            if (churn === "appearing") {
              const pid = 201 + scans;
              mkdirSync(join(root, String(pid)));
              writeFileSync(
                join(root, String(pid), "stat"),
                `${pid} (other) S 1 88 0`,
              );
            }
            return readdirSync(root);
          },
          readStat: (path) => {
            // The PID exited after enumeration, before its stat could be read.
            if (churn === "exiting" && path === join(f.root, "201", "stat")) {
              rmSync(join(f.root, "201"), { recursive: true });
            }
            return readFileSync(path, "utf8");
          },
        }),
        false,
      );
    } finally {
      f.clean();
    }
  },
);

test("an unrelated exiting process with pgrp -1 does not keep a zombie group alive", () => {
  const f = fixture([
    [101, 77, "Z"],
    [102, 77, "Z"],
    [7998, -1, "X"],
  ]);
  try {
    // Shape reported from Linux /proc while this unrelated process exited.
    writeFileSync(
      join(f.root, "7998", "stat"),
      "7998 (python3) X 0 -1 -1 0 -1 4227148 62 0 0 0 0 0 0 0 20 0",
    );
    assert.equal(groupAlive(77, f.options), false);
  } finally {
    f.clean();
  }
});

test.each([
  "201 (worker) X 1 not-a-group 0",
  "201 (worker) X 1 77.5 0",
  "201 (worker) X 1 9007199254740992 0",
  "201 (worker) X 1 -9007199254740992 0",
  "201 (worker) ? 1 -1 0",
  "999 (worker) X 1 -1 0",
])("uncertain unrelated process metadata prevents cleanup: %s", (stat) => {
  const f = fixture([
    [101, 77, "Z"],
    [201, 88, "S"],
  ]);
  try {
    writeFileSync(join(f.root, "201", "stat"), stat);
    assert.equal(groupAlive(77, f.options), true);
  } finally {
    f.clean();
  }
});

test.each([
  ["ENOENT", false],
  ["ESRCH", false],
  ["EACCES", true],
  ["EPERM", true],
  ["EIO", true],
])(
  "a stat read failing with %s preserves the cleanup boundary",
  (code, alive) => {
    const f = fixture([
      [101, 77, "Z"],
      [201, 88, "S"],
    ]);
    try {
      assert.equal(
        groupAlive(77, {
          ...f.options,
          readStat: (path) => {
            if (path === join(f.root, "201", "stat")) {
              throw Object.assign(new Error("stat unavailable"), { code });
            }
            return readFileSync(path, "utf8");
          },
        }),
        alive,
      );
    } finally {
      f.clean();
    }
  },
);

test.each([
  "joining live",
  "joining zombie",
  "exiting",
  "losing its group",
  "reused for a live process",
])("a target group member %s prevents cleanup", (change) => {
  const f = fixture([
    [100, 77, "Z"],
    [101, 77, "Z"],
  ]);
  let scans = 0;
  try {
    assert.equal(
      groupAlive(77, {
        ...f.options,
        readDirectory: (root) => {
          if (++scans === 2) {
            if (change.startsWith("joining")) {
              mkdirSync(join(root, "102"));
              const state = change === "joining live" ? "S" : "Z";
              writeFileSync(
                join(root, "102", "stat"),
                `102 (child) ${state} 1 77 0`,
              );
            } else if (change === "exiting") {
              rmSync(join(root, "101"), { recursive: true });
            } else if (change === "losing its group") {
              writeFileSync(join(root, "101", "stat"), "101 (worker) X 1 -1 0");
            } else {
              writeFileSync(join(root, "101", "stat"), "101 (worker) S 1 77 0");
            }
          }
          return readdirSync(root);
        },
      }),
      true,
    );
  } finally {
    f.clean();
  }
});

test("non-Linux keeps signal-zero semantics; a missing group is stopped", () => {
  let calls = 0;
  assert.equal(
    groupAlive(77, {
      platform: "darwin",
      signal: (pid, value) => {
        assert.equal(pid, -77);
        assert.equal(value, 0);
        calls++;
      },
      readDirectory: () => {
        throw new Error("must not scan");
      },
    }),
    true,
  );
  assert.equal(calls, 1);
  assert.equal(
    groupAlive(77, {
      signal: () => {
        throw Object.assign(new Error(), { code: "ESRCH" });
      },
    }),
    false,
  );
});

test("malformed process metadata cannot authorize cleanup", () => {
  const f = fixture([[101, 77, "Z"]]);
  try {
    writeFileSync(join(f.root, "101", "stat"), "truncated");
    assert.equal(groupAlive(77, f.options), true);
  } finally {
    f.clean();
  }
});

test("private purge and claim release require a stopped group and every port closed", () => {
  const ports = { ingress: "closed", backend: "closed" };
  assert.equal(privatePurgeAllowed(true, ports), true);
  assert.equal(privatePurgeAllowed(false, ports), false);
  assert.equal(
    privatePurgeAllowed(true, { ...ports, backend: "still open" }),
    false,
  );
  assert.equal(privatePurgeAllowed(true, {}), false);
});
