import { describe, it, expect } from "vitest";
import {
  inferKindFromHost,
  isLocalAddress,
  isValidHostUrl,
  normalizeHostInput,
  stripTrailingSlashes,
} from "#/api/backend-registry/backend-host";

describe("stripTrailingSlashes", () => {
  it("removes one or more trailing slashes", () => {
    expect(stripTrailingSlashes("https://example.com/")).toBe(
      "https://example.com",
    );
    expect(stripTrailingSlashes("https://example.com///")).toBe(
      "https://example.com",
    );
  });

  it("leaves a host without trailing slashes untouched", () => {
    expect(stripTrailingSlashes("https://example.com")).toBe(
      "https://example.com",
    );
    expect(stripTrailingSlashes("")).toBe("");
  });

  it("does not strip slashes that are not at the end", () => {
    expect(stripTrailingSlashes("https://example.com/base")).toBe(
      "https://example.com/base",
    );
  });
});

describe("isLocalAddress", () => {
  it("treats loopback and any-address forms as local", () => {
    for (const host of ["localhost", "LOCALHOST", "::1", "::", "0.0.0.0"]) {
      expect(isLocalAddress(host)).toBe(true);
    }
  });

  it("treats the 127.0.0.0/8 range and IPv4-mapped loopback as local", () => {
    expect(isLocalAddress("127.0.0.1")).toBe(true);
    expect(isLocalAddress("127.99.1.2")).toBe(true);
    expect(isLocalAddress("::ffff:127.0.0.1")).toBe(true);
  });

  it("treats RFC 1918 private ranges as local", () => {
    expect(isLocalAddress("10.0.0.5")).toBe(true);
    expect(isLocalAddress("192.168.1.5")).toBe(true);
    expect(isLocalAddress("172.16.0.1")).toBe(true);
    expect(isLocalAddress("172.31.255.254")).toBe(true);
  });

  it("does not treat neighbours of the private ranges as local", () => {
    expect(isLocalAddress("172.15.0.1")).toBe(false);
    expect(isLocalAddress("172.32.0.1")).toBe(false);
    expect(isLocalAddress("11.0.0.1")).toBe(false);
    expect(isLocalAddress("192.169.1.1")).toBe(false);
  });

  it("treats IPv6 link-local and unique-local prefixes as local", () => {
    expect(isLocalAddress("fe80::1")).toBe(true);
    expect(isLocalAddress("fd00::1")).toBe(true);
    expect(isLocalAddress("[fe80::1]")).toBe(true);
  });

  it("treats .local and single-label hostnames as local", () => {
    expect(isLocalAddress("myserver.local")).toBe(true);
    expect(isLocalAddress("nas")).toBe(true);
  });

  it("treats public hostnames as non-local", () => {
    expect(isLocalAddress("example.com")).toBe(false);
    expect(isLocalAddress("app.all-hands.dev")).toBe(false);
    expect(isLocalAddress("2001:4860:4860::8888")).toBe(false);
  });
});

describe("normalizeHostInput", () => {
  it("returns an empty string for empty and whitespace-only input", () => {
    expect(normalizeHostInput("")).toBe("");
    expect(normalizeHostInput("   ")).toBe("");
  });

  it("trims surrounding whitespace", () => {
    expect(normalizeHostInput("  example.com  ")).toBe("https://example.com");
  });

  it("respects an explicit scheme and strips trailing slashes", () => {
    expect(normalizeHostInput("http://localhost:8000")).toBe(
      "http://localhost:8000",
    );
    expect(normalizeHostInput("https://example.com/")).toBe(
      "https://example.com",
    );
    // Scheme matching is case-insensitive.
    expect(normalizeHostInput("HTTPS://example.com")).toBe(
      "HTTPS://example.com",
    );
  });

  it("prepends https:// to a bare public host", () => {
    expect(normalizeHostInput("example.com")).toBe("https://example.com");
    expect(normalizeHostInput("example.com:9000")).toBe(
      "https://example.com:9000",
    );
  });

  it("prepends http:// to bare localhost and private addresses", () => {
    expect(normalizeHostInput("localhost:8000")).toBe("http://localhost:8000");
    expect(normalizeHostInput("192.168.1.5:8000")).toBe(
      "http://192.168.1.5:8000",
    );
    expect(normalizeHostInput("10.0.0.5")).toBe("http://10.0.0.5");
  });

  it("prepends http:// to bracketed IPv6 loopback, keeping the brackets", () => {
    expect(normalizeHostInput("[::1]:8080")).toBe("http://[::1]:8080");
    expect(normalizeHostInput("[fe80::1]")).toBe("http://[fe80::1]");
  });

  it("prepends https:// to a bracketed public IPv6 address", () => {
    expect(normalizeHostInput("[2001:4860:4860::8888]:443")).toBe(
      "https://[2001:4860:4860::8888]:443",
    );
  });

  it("treats an unbracketed IPv6 address as a whole hostname", () => {
    // More than one colon and no brackets means the entire string is the
    // hostname rather than a host:port pair.
    expect(normalizeHostInput("::1")).toBe("http://::1");
  });

  it("prepends http:// to .local and single-label hosts", () => {
    expect(normalizeHostInput("myserver.local")).toBe("http://myserver.local");
    expect(normalizeHostInput("nas:8000")).toBe("http://nas:8000");
  });
});

describe("isValidHostUrl", () => {
  it("rejects empty and whitespace-only input", () => {
    expect(isValidHostUrl("")).toBe(false);
    expect(isValidHostUrl("   ")).toBe(false);
  });

  it("rejects input containing whitespace", () => {
    expect(isValidHostUrl("my host")).toBe(false);
    expect(isValidHostUrl("http://example.com/a b")).toBe(false);
  });

  it("accepts bare hosts, with and without a port", () => {
    expect(isValidHostUrl("example.com")).toBe(true);
    expect(isValidHostUrl("example.com:9000")).toBe(true);
    expect(isValidHostUrl("  example.com  ")).toBe(true);
  });

  it("accepts localhost and private addresses", () => {
    expect(isValidHostUrl("localhost:8000")).toBe(true);
    expect(isValidHostUrl("127.0.0.1")).toBe(true);
    expect(isValidHostUrl("192.168.1.5:8000")).toBe(true);
  });

  it("accepts .local hostnames", () => {
    expect(isValidHostUrl("myserver.local")).toBe(true);
  });

  it("accepts explicit http and https URLs", () => {
    expect(isValidHostUrl("http://localhost:8000")).toBe(true);
    expect(isValidHostUrl("https://app.all-hands.dev")).toBe(true);
  });

  it("accepts bracketed IPv6 but not the unbracketed form", () => {
    expect(isValidHostUrl("[::1]:8080")).toBe(true);
    // Unbracketed IPv6 normalises to `http://::1`, which the URL parser
    // rejects because a bare IPv6 host must be bracketed. Pinned here as
    // existing behaviour: the form requires the bracketed spelling.
    expect(isValidHostUrl("::1")).toBe(false);
  });
});

describe("inferKindFromHost", () => {
  it("infers cloud for OpenHands Cloud hosts", () => {
    expect(inferKindFromHost("https://app.all-hands.dev")).toBe("cloud");
    expect(inferKindFromHost("app.all-hands.dev")).toBe("cloud");
    expect(inferKindFromHost("https://app.openhands.dev")).toBe("cloud");
  });

  it("infers local for agent-server and custom hosts", () => {
    expect(inferKindFromHost("http://localhost:8000")).toBe("local");
    expect(inferKindFromHost("https://example.com")).toBe("local");
  });

  it("does not misread a look-alike domain as cloud", () => {
    // Suffix matching, not substring matching.
    expect(inferKindFromHost("https://malicious-all-hands.dev")).toBe("local");
    expect(inferKindFromHost("https://all-hands.dev.evil.com")).toBe("local");
  });

  it("infers local for an empty or unparseable host", () => {
    expect(inferKindFromHost("")).toBe("local");
    expect(inferKindFromHost("not-a-url")).toBe("local");
  });
});
