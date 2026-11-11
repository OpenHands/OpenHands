import { afterEach, describe, expect, it } from "vitest";
import { resetMockMetaProfiles } from "./meta-profiles-handlers";

const endpoint = "http://127.0.0.1:8000/api/meta-profiles";

async function read(name: string) {
  return fetch(`${endpoint}/${name}`);
}

afterEach(resetMockMetaProfiles);

describe("Model Router MSW handlers", () => {
  it("lists, saves, loads, activates, and deletes shared routers", async () => {
    const empty = await fetch(endpoint);
    expect(await empty.json()).toEqual({
      meta_profiles: [],
      active_meta_profile: null,
    });

    const config = {
      classifier_model: "studio-llm",
      prompt_template: "Route {{ instance_text }}",
      model_table: "studio-llm",
    };
    const saved = await fetch(`${endpoint}/balanced`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    expect(saved.ok).toBe(true);

    const detail = await read("balanced");
    expect(await detail.json()).toEqual({ name: "balanced", config });

    const list = await fetch(endpoint);
    expect(await list.json()).toEqual({
      meta_profiles: [
        { name: "balanced", classifier_model: "studio-llm", num_classes: 0 },
      ],
      active_meta_profile: null,
    });

    const activated = await fetch(`${endpoint}/balanced/activate`, {
      method: "POST",
    });
    expect(activated.ok).toBe(true);
    const activeList = await fetch(endpoint);
    expect(await activeList.json()).toMatchObject({
      active_meta_profile: "balanced",
    });

    const deleted = await fetch(`${endpoint}/balanced`, { method: "DELETE" });
    expect(deleted.ok).toBe(true);
    const cleared = await fetch(endpoint);
    expect(await cleared.json()).toEqual({
      meta_profiles: [],
      active_meta_profile: null,
    });
  });

  it("reports a missing router and rejects an invalid router configuration", async () => {
    expect((await read("missing")).status).toBe(404);
    const invalid = await fetch(`${endpoint}/bad`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ classifier_model: "" }),
    });
    expect(invalid.status).toBe(422);
    expect((await fetch(endpoint)).ok).toBe(true);
  });
});
