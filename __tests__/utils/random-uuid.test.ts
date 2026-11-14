import { afterEach, describe, expect, it, vi } from "vitest";
import { randomUUID } from "#/utils/random-uuid";
import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import { useSessionActivityStore } from "#/stores/session-activity-store";

const UUID_V4 =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

// Plain-HTTP non-loopback origins are not secure contexts: browsers expose
// crypto.getRandomValues there but leave crypto.randomUUID undefined.
function stubNonSecureContextCrypto() {
  const realCrypto = globalThis.crypto;
  const getRandomValues = vi.fn((array: Uint8Array<ArrayBuffer>) =>
    realCrypto.getRandomValues(array),
  );
  vi.stubGlobal("crypto", { getRandomValues });
  return getRandomValues;
}

afterEach(() => {
  vi.unstubAllGlobals();
  useConversationLaunchStore.setState({ launches: {} });
  useSessionActivityStore.setState({ entries: {} });
});

describe("randomUUID", () => {
  it("uses crypto.randomUUID when the context provides it", () => {
    const native = vi
      .spyOn(globalThis.crypto, "randomUUID")
      .mockReturnValue("11111111-2222-4333-8444-555555555555");

    expect(randomUUID()).toBe("11111111-2222-4333-8444-555555555555");
    expect(native).toHaveBeenCalledOnce();
    native.mockRestore();
  });

  it("returns distinct UUIDv4 values from getRandomValues outside a secure context", () => {
    const getRandomValues = stubNonSecureContextCrypto();

    const ids = [randomUUID(), randomUUID(), randomUUID()];

    ids.forEach((id) => expect(id).toMatch(UUID_V4));
    expect(new Set(ids).size).toBe(ids.length);
    expect(getRandomValues).toHaveBeenCalledTimes(ids.length);
  });
});

describe("conversation launch outside a secure context", () => {
  it("records the launch and its session activity without throwing", () => {
    stubNonSecureContextCrypto();

    const id = useConversationLaunchStore.getState().begin({
      backendId: "cloud",
      orgId: "org",
      message: "Say hello",
      files: [],
      attachments: {
        content: "Say hello",
        images: [],
        files: [],
        imagesMarkedUploadAsFile: [],
      },
    });

    expect(useConversationLaunchStore.getState().launches[id]).toMatchObject({
      message: "Say hello",
    });
    const [entry] = Object.values(
      useSessionActivityStore.getState().entries,
    )[0];
    expect(entry).toMatchObject({ kind: "initialize", phase: "pending" });
    expect(entry.id).toMatch(UUID_V4);
  });
});
