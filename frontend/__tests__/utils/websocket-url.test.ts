import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  extractBaseHost,
  extractPathPrefix,
  buildHttpBaseUrl,
  buildWebSocketUrl,
} from "@/utils/websocket-url";

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

  describe("extractBaseHost", () => {
    it("should extract host from a standard URL", () => {
      const result = extractBaseHost(
        "https://example.com/api/conversations/123",
      );
      expect(result).toBe("example.com");
    });

    it("should extract host with port from URL", () => {
      const result = extractBaseHost(
        "http://localhost:3000/api/conversations/123",
      );
      expect(result).toBe("localhost:3000");
    });

    it("should extract host from proxy deployment URL", () => {
      const result = extractBaseHost(
        "https://openhands.example.com/runtime/55313/api/conversations/abc123",
      );
      expect(result).toBe("openhands.example.com");
    });

    it("should return window.location.host for relative URLs", () => {
      const result = extractBaseHost("/api/conversations/123");
      expect(result).toBe("localhost:3001");
    });

    it("should return window.location.host for null", () => {
      const result = extractBaseHost(null);
      expect(result).toBe("localhost:3001");
    });

    it("should return window.location.host for undefined", () => {
      const result = extractBaseHost(undefined);
      expect(result).toBe("localhost:3001");
    });

    it("should return window.location.host for invalid URL", () => {
      const result = extractBaseHost("not-a-valid-url");
      expect(result).toBe("localhost:3001");
    });
  });

  describe("extractPathPrefix", () => {
    it("should return empty string for URL without path prefix", () => {
      const result = extractPathPrefix(
        "https://example.com/api/conversations/123",
      );
      expect(result).toBe("");
    });

    it("should extract path prefix from proxy deployment URL", () => {
      const result = extractPathPrefix(
        "https://openhands.example.com/runtime/55313/api/conversations/abc123",
      );
      expect(result).toBe("/runtime/55313");
    });

    it("should handle multiple path segments before /api/conversations", () => {
      const result = extractPathPrefix(
        "https://example.com/prefix/sub/path/api/conversations/123",
      );
      expect(result).toBe("/prefix/sub/path");
    });

    it("should remove trailing slash from path prefix", () => {
      // This test ensures the function handles URLs where the path ends with /
      const result = extractPathPrefix(
        "https://example.com/runtime/55313/api/conversations/123",
      );
      expect(result).not.toMatch(/\/$/);
    });

    it("should return empty string for relative URLs", () => {
      const result = extractPathPrefix("/api/conversations/123");
      expect(result).toBe("");
    });

    it("should return empty string for null", () => {
      const result = extractPathPrefix(null);
      expect(result).toBe("");
    });

    it("should return empty string for undefined", () => {
      const result = extractPathPrefix(undefined);
      expect(result).toBe("");
    });

    it("should return empty string for invalid URL", () => {
      const result = extractPathPrefix("not-a-valid-url");
      expect(result).toBe("");
    });

    it("should handle URL with only root path before /api/conversations", () => {
      const result = extractPathPrefix(
        "https://example.com/api/conversations/123",
      );
      expect(result).toBe("");
    });
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
      window.location.protocol = "http:";
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
    it("should return null when conversationId is undefined", () => {
      const result = buildWebSocketUrl(
        undefined,
        "https://example.com/api/conversations/123",
      );
      expect(result).toBeNull();
    });

    it("should return null when conversationId is empty string", () => {
      const result = buildWebSocketUrl(
        "",
        "https://example.com/api/conversations/123",
      );
      expect(result).toBeNull();
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
      window.location.protocol = "http:";
      const result = buildWebSocketUrl(
        "conv-123",
        "http://localhost:3000/api/conversations/conv-123",
      );
      expect(result).toBe("ws://localhost:3000/sockets/events/conv-123");
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
