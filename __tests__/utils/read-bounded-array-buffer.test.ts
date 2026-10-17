import { describe, expect, it, vi } from "vitest";
import { readBoundedArrayBuffer } from "#/utils/ooxml-preview";

const streamOf = (...chunks: Uint8Array[]): ReadableStream<Uint8Array> =>
  new ReadableStream<Uint8Array>({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(chunk));
      controller.close();
    },
  });

describe("readBoundedArrayBuffer", () => {
  it("rejects from the declared content-length without reading the body", async () => {
    const cancel = vi.fn().mockResolvedValue(undefined);
    const response = {
      headers: new Headers({ "content-length": "5000" }),
      body: { cancel },
      arrayBuffer: async () => new ArrayBuffer(10),
    } as unknown as Response;

    await expect(readBoundedArrayBuffer(response, 1000)).rejects.toThrow(
      /exceeds 1000 bytes/,
    );
    // The early reject never touches the stream, so it must be cancelled or the
    // browser keeps downloading a file the caller will never parse.
    expect(cancel).toHaveBeenCalled();
  });

  it("rejects a body that outgrows the cap while streaming", async () => {
    // No declared length, and the body is served in chunks: the ceiling must be
    // enforced as bytes arrive, not after buffering the whole thing.
    const response = {
      headers: new Headers(),
      body: streamOf(new Uint8Array(600), new Uint8Array(600)),
      arrayBuffer: async () => new ArrayBuffer(1200),
    } as unknown as Response;

    await expect(readBoundedArrayBuffer(response, 1000)).rejects.toThrow(
      /exceeds 1000 bytes/,
    );
  });

  it("returns the bytes when they fit", async () => {
    const response = {
      headers: new Headers(),
      body: streamOf(new Uint8Array([1, 2]), new Uint8Array([3])),
      arrayBuffer: async () => new ArrayBuffer(0),
    } as unknown as Response;

    const buffer = await readBoundedArrayBuffer(response, 1000);

    expect(Array.from(new Uint8Array(buffer))).toEqual([1, 2, 3]);
  });

  it("falls back to arrayBuffer() when the response has no readable stream", async () => {
    const response = {
      headers: new Headers(),
      arrayBuffer: async () => new Uint8Array([9, 9]).buffer,
    } as unknown as Response;

    const buffer = await readBoundedArrayBuffer(response, 1000);

    expect(Array.from(new Uint8Array(buffer))).toEqual([9, 9]);
  });
});
