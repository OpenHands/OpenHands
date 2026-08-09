import { describe, expect, it, vi } from "vitest";
import {
  buildDesktopPortOverrides,
  formatPortRemapMessage,
  getDesktopPreferredPorts,
  planDesktopPortRemap,
} from "../../scripts/desktop-port-remap.mjs";

const DEFAULTS = {
  ports: {
    proxy: 8000,
    agentServer: 18000,
    automation: 18001,
  },
};

describe("getDesktopPreferredPorts", () => {
  it("uses shared defaults when env overrides are absent", () => {
    // Arrange / Act
    const ports = getDesktopPreferredPorts(DEFAULTS, {});

    // Assert
    expect(ports.map((p) => [p.name, p.preferred, p.envKey])).toEqual([
      ["ingress", 8000, "PORT"],
      ["agent-server", 18000, "OH_CANVAS_SAFE_BACKEND_PORT"],
      ["automation", 18001, "OH_CANVAS_SAFE_AUTOMATION_PORT"],
      ["frontend", 3001, "OH_CANVAS_SAFE_VITE_PORT"],
    ]);
  });

  it("honors env overrides for each service port", () => {
    // Arrange / Act
    const ports = getDesktopPreferredPorts(DEFAULTS, {
      PORT: "9000",
      OH_CANVAS_SAFE_BACKEND_PORT: "19000",
      OH_CANVAS_SAFE_AUTOMATION_PORT: "19001",
      OH_CANVAS_SAFE_VITE_PORT: "3101",
    });

    // Assert
    expect(ports.find((p) => p.name === "ingress")?.preferred).toBe(9000);
    expect(ports.find((p) => p.name === "agent-server")?.preferred).toBe(19000);
    expect(ports.find((p) => p.name === "automation")?.preferred).toBe(19001);
    expect(ports.find((p) => p.name === "frontend")?.preferred).toBe(3101);
  });
});

describe("buildDesktopPortOverrides", () => {
  const preferred = getDesktopPreferredPorts(DEFAULTS, {});

  it("returns no remaps when every preferred port is free", () => {
    // Arrange
    const allocated = {
      ingress: 8000,
      "agent-server": 18000,
      automation: 18001,
      frontend: 3001,
    };

    // Act
    const result = buildDesktopPortOverrides(preferred, allocated);

    // Assert
    expect(result.remaps).toEqual([]);
    expect(result.env).toEqual({});
  });

  it("builds env overrides only for ports that had to move", () => {
    // Arrange
    const allocated = {
      ingress: 8000,
      "agent-server": 54321,
      automation: 54322,
      frontend: 3001,
    };

    // Act
    const result = buildDesktopPortOverrides(preferred, allocated);

    // Assert
    expect(result.remaps).toEqual([
      {
        name: "agent-server",
        label: "agent-server",
        preferred: 18000,
        actual: 54321,
      },
      {
        name: "automation",
        label: "automation",
        preferred: 18001,
        actual: 54322,
      },
    ]);
    expect(result.env).toEqual({
      OH_CANVAS_SAFE_BACKEND_PORT: "54321",
      OH_CANVAS_SAFE_AUTOMATION_PORT: "54322",
    });
  });

  it("throws when an allocated port is missing", () => {
    // Arrange / Act / Assert
    expect(() =>
      buildDesktopPortOverrides(preferred, {
        ingress: 8000,
        "agent-server": 18000,
        automation: 18001,
        // frontend missing
      }),
    ).toThrow(/frontend/i);
  });
});

describe("formatPortRemapMessage", () => {
  it("lists each busy port and its suggested replacement", () => {
    // Arrange / Act
    const message = formatPortRemapMessage([
      {
        label: "agent-server",
        preferred: 18000,
        actual: 54321,
      },
      {
        label: "automation",
        preferred: 18001,
        actual: 54322,
      },
    ]);

    // Assert
    expect(message).toContain("agent-server: 18000 → 54321");
    expect(message).toContain("automation: 18001 → 54322");
    expect(message).toMatch(/alternate ports/i);
  });
});

describe("planDesktopPortRemap", () => {
  it("delegates to findFreePorts and surfaces remaps", async () => {
    // Arrange
    const preferred = getDesktopPreferredPorts(DEFAULTS, {});
    const findFreePorts = vi.fn().mockResolvedValue({
      ingress: 8000,
      "agent-server": 54321,
      automation: 18001,
      frontend: 3001,
    });

    // Act
    const plan = await planDesktopPortRemap(preferred, findFreePorts);

    // Assert
    expect(findFreePorts).toHaveBeenCalledWith([
      { name: "ingress", preferred: 8000 },
      { name: "agent-server", preferred: 18000 },
      { name: "automation", preferred: 18001 },
      { name: "frontend", preferred: 3001 },
    ]);
    expect(plan.remaps).toHaveLength(1);
    expect(plan.remaps[0]).toMatchObject({
      name: "agent-server",
      preferred: 18000,
      actual: 54321,
    });
    expect(plan.env).toEqual({ OH_CANVAS_SAFE_BACKEND_PORT: "54321" });
  });
});
