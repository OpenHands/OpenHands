import { describe, expect, it } from "vitest";
import { collectRunGraphChildren } from "#/routes/conversation-graph";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  FACTORY_PLAN_WORKSTREAM_ID,
  FACTORY_RUN_ID_TAG_KEY,
  FACTORY_RUN_TAG_KEY,
  FACTORY_WORKSTREAM_ID_TAG_KEY,
} from "#/api/agent-server-adapter";

function conv(
  id: string,
  tags: Record<string, string> | undefined,
  subConversationIds: string[] = [],
) {
  return {
    id,
    tags,
    sub_conversation_ids: subConversationIds,
  } as unknown as AppConversation;
}

function factoryTags(runId: string, workstreamId: string) {
  return {
    [FACTORY_RUN_TAG_KEY]: "1",
    [FACTORY_RUN_ID_TAG_KEY]: runId,
    [FACTORY_WORKSTREAM_ID_TAG_KEY]: workstreamId,
  };
}

describe("collectRunGraphChildren", () => {
  it("includes native sub-conversation children", () => {
    const parent = conv("p", undefined, ["n1"]);
    const native = [conv("n1", undefined)];
    expect(collectRunGraphChildren(parent, native, [])).toEqual([native[0]]);
  });

  it("includes factory workstreams sharing the run, excluding the plan", () => {
    const parent = conv("p", factoryTags("run-1", FACTORY_PLAN_WORKSTREAM_ID));
    const siblings = [
      conv("w1", factoryTags("run-1", "ws-1")),
      conv("w2", factoryTags("run-1", "ws-2")),
      // Different run — must not appear.
      conv("w3", factoryTags("run-2", "ws-1")),
    ];
    const ids = collectRunGraphChildren(parent, [], siblings).map((c) => c.id);
    expect(ids.sort()).toEqual(["w1", "w2"]);
  });

  it("excludes the parent itself and dedupes shared ids", () => {
    const parent = conv("p", factoryTags("run-1", FACTORY_PLAN_WORKSTREAM_ID), [
      "w1",
    ]);
    const native = [conv("w1", undefined)];
    const siblings = [
      conv("p", factoryTags("run-1", FACTORY_PLAN_WORKSTREAM_ID)),
      conv("w1", factoryTags("run-1", "ws-1")),
    ];
    const children = collectRunGraphChildren(parent, native, siblings);
    expect(children).toHaveLength(1);
    expect(children[0].id).toBe("w1");
  });

  it("returns native children only for a non-factory parent", () => {
    const parent = conv("p", { origin: "slack" }, ["n1"]);
    const native = [conv("n1", undefined)];
    const unrelated = [conv("other", factoryTags("run-9", "ws-1"))];
    expect(collectRunGraphChildren(parent, native, unrelated)).toEqual([
      native[0],
    ]);
  });

  it("returns nothing for a missing parent", () => {
    expect(collectRunGraphChildren(undefined, [], [])).toEqual([]);
  });
});
