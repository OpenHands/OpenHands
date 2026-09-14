import { describe, expect, it } from "vitest";

import { chooseWorkerUrl } from "#/hooks/query/use-forwarded-preview-url";

describe("chooseWorkerUrl", () => {
  it("selects the first browser-reachable WORKER URL in stable name order", () => {
    expect(
      chooseWorkerUrl([
        { name: "VSCODE", url: "https://editor.example.test" },
        { name: "WORKER_5174", url: "https://app.example.test" },
        { name: "WORKER_3000", url: "https://app-old.example.test" },
      ]),
    ).toBe("https://app-old.example.test");
  });

  it("ignores non-worker entries and malformed URLs", () => {
    expect(
      chooseWorkerUrl([
        { name: "AGENT_SERVER", url: "https://agent.example.test" },
        { name: "WORKER_5173", url: "localhost:5173" },
      ]),
    ).toBeNull();
  });

  it("returns null while the sandbox has not exposed a worker", () => {
    expect(chooseWorkerUrl(null)).toBeNull();
    expect(chooseWorkerUrl([])).toBeNull();
  });
});
