// @vitest-environment node
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { expect, it } from "vitest";
import { findMissingQueryReturns } from "../../scripts/check-query-returns.mjs";

it("rejects missing query data, including promised and union returns", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "canvas-query-returns-"));
  try {
    const config = path.join(dir, "tsconfig.json");
    fs.writeFileSync(
      config,
      JSON.stringify({
        compilerOptions: { strict: true },
        files: ["queries.ts"],
      }),
    );
    fs.writeFileSync(
      path.join(dir, "queries.ts"),
      `
import type {} from "@tanstack/react-query";
const missing = () => {};
const plain = { queryFn: missing };
const promised = { queryFn: async () => undefined };
const union = { queryFn: (): Promise<number | undefined> => Promise.resolve(1) };
const valid = { queryFn: async () => null };
const queryFn = () => {};
const shorthand = { queryFn };
const method = { queryFn() {} };
`,
    );
    expect(
      findMissingQueryReturns(config, dir).map(
        (message) => message.match(/queries\.ts:(\d+):/)?.[1],
      ),
    ).toEqual(["4", "5", "6", "9", "10"]);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
