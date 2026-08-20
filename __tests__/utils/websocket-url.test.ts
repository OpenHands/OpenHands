import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { buildHttpBaseUrl, buildWebSocketUrl } from "#/utils/websocket-url";

describe("websocket-url utilities", () => {
  beforeEach(() => {
    vi.stubGlobal("location", {
      host: "localhost:3001",
      protocol: "https:",
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("buildHttpBaseUrl", () => {
    it("should build HTTP URL without path prefix", () => {
      const result = buildHttpBaseUrl(
        "https://example.com/api/conversations/123",
      );
      expect(result).toBe("https://example.com");
    });

    it("should build HTTP URL with path prefix for proxy deployment", () => {
      const result = buildHttpBaseUrl(
        "https://openhands.example.com/runtime/55313/api/conversations/abc123",
      );
      expect(result).toBe("https://openhands.example.com/runtime/55313");
    });

    it("should use http protocol when window.location.protocol is http:", () => {
      vi.stubGlobal("location", {
        host: "localhost:3001",
        protocol: "http:",
      });

      const result = buildHttpBaseUrl(
        "http://localhost:3000/api/conversations/123",
      );
      expect(result).toBe("http://localhost:3000");
    });

    it("should fallback to window.location for null URL", () => {
      const result = buildHttpBaseUrl(null);
      expect(result).toBe("https://localhost:3001");
    });
  });

  describe("buildWebSocketUrl", () => {
    it("should return null when conversationId is undefined or empty", () => {
      expect(
        buildWebSocketUrl(
          undefined,
          "https://example.com/api/conversations/123",
        ),
      ).toBeNull();
      expect(
        buildWebSocketUrl("", "https://example.com/api/conversations/123"),
      ).toBeNull();
    });

    it("should build WebSocket URL without path prefix", () => {
      const result = buildWebSocketUrl(
        "conv-123",
        "https://example.com/api/conversations/conv-123",
      );
      expect(result).toBe("wss://example.com/sockets/events/conv-123");
    });

    it("should build WebSocket URL with path prefix for proxy deployment", () => {
      const result = buildWebSocketUrl(
        "abc123",
        "https://openhands.example.com/runtime/55313/api/conversations/abc123",
      );
      expect(result).toBe(
        "wss://openhands.example.com/runtime/55313/sockets/events/abc123",
      );
    });

    it("should use ws protocol when window.location.protocol is http:", () => {
      vi.stubGlobal("location", {
        host: "localhost:3001",
        protocol: "http:",
      });

      const result = buildWebSocketUrl(
        "conv-123",
        "http://localhost:3000/api/conversations/conv-123",
      );
      expect(result).toBe("ws://localhost:3000/sockets/events/conv-123");
    });

    it("should use ws for a remote HTTP page served by the local ingress", () => {
      vi.stubGlobal("location", {
        host: "spark-1874.tailae62af.ts.net:8000",
        hostname: "spark-1874.tailae62af.ts.net",
        protocol: "http:",
      });

      const result = buildWebSocketUrl(
        "conv-123",
        "http://localhost:8000/api/conversations/conv-123",
      );
      expect(result).toBe(
        "ws://spark-1874.tailae62af.ts.net:8000/sockets/events/conv-123",
      );
    });

    it("should fallback to window.location.host for null URL", () => {
      const result = buildWebSocketUrl("conv-123", null);
      expect(result).toBe("wss://localhost:3001/sockets/events/conv-123");
    });

    it("should handle complex path prefixes", () => {
      const result = buildWebSocketUrl(
        "test-conv",
        "https://app.example.com/org/team/runtime/12345/api/conversations/test-conv",
      );
      expect(result).toBe(
        "wss://app.example.com/org/team/runtime/12345/sockets/events/test-conv",
      );
    });
  });
});
