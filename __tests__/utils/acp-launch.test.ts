import { describe, expect, it } from "vitest";
import type { ACPAgentProfile } from "@openhands/typescript-client";
import { ACP_PROVIDERS, getAcpProvider } from "#/constants/acp-providers";
import {
  acpLaunchArgv,
  acpLaunchFromCommandText,
  acpLaunchFromProfile,
  detectAcpPreset,
  isAcpLaunchDiscoverable,
} from "#/utils/acp-launch";

type LaunchProfile = Pick<
  ACPAgentProfile,
  "acp_server" | "acp_command" | "acp_args"
>;

const profile = (overrides: Partial<LaunchProfile>): LaunchProfile => ({
  acp_server: "custom",
  acp_command: null,
  acp_args: null,
  ...overrides,
});

describe("acpLaunchFromProfile", () => {
  it("parses a saved command and keeps API-set args separate", () => {
    expect(
      acpLaunchFromProfile(
        profile({
          acp_command: "my-acp-wrapper --flag 'two words'",
          acp_args: ["--extra"],
        }),
      ),
    ).toEqual({
      server: "custom",
      command: ["my-acp-wrapper", "--flag", "two words"],
      args: ["--extra"],
    });
  });

  it("leaves a built-in provider's command to the server default", () => {
    expect(
      acpLaunchFromProfile(profile({ acp_server: "claude-code" })),
    ).toEqual({ server: "claude-code", command: [], args: [] });
  });

  it("ignores an explicit built-in command so its payload stays unchanged", () => {
    const defaultCommand = getAcpProvider("claude-code")!.default_command;
    // The profiles API can store the registry default explicitly; Settings
    // stores null. Both must ask exactly like a profile without a command.
    expect(
      acpLaunchFromProfile(
        profile({
          acp_server: "claude-code",
          acp_command: defaultCommand.join(" "),
        }),
      ),
    ).toEqual({ server: "claude-code", command: [], args: [] });
    // A customized built-in command is out of scope for discovery, not a
    // reason to change the payload or the remembered-list scope.
    expect(
      acpLaunchFromProfile(
        profile({
          acp_server: "claude-code",
          acp_command: "my-claude-wrapper --flag",
          acp_args: ["--extra"],
        }),
      ),
    ).toEqual({ server: "claude-code", command: [], args: [] });
  });

  it("returns null without a server", () => {
    expect(acpLaunchFromProfile(null)).toBeNull();
    expect(acpLaunchFromProfile(undefined)).toBeNull();
  });
});

describe("detectAcpPreset", () => {
  it("matches a provider on its tokenized command", () => {
    const claude = getAcpProvider("claude-code")!;
    expect(
      detectAcpPreset(claude.default_command.join(" "), ACP_PROVIDERS),
    ).toBe("claude-code");
    expect(
      detectAcpPreset(
        `  ${claude.default_command.join("   ")}  `,
        ACP_PROVIDERS,
      ),
    ).toBe("claude-code");
    expect(detectAcpPreset("my-acp-wrapper", ACP_PROVIDERS)).toBe("custom");
    expect(detectAcpPreset("", ACP_PROVIDERS)).toBe("custom");
  });
});

describe("acpLaunchFromCommandText", () => {
  it("asks a built-in provider without an explicit command", () => {
    const claude = getAcpProvider("claude-code")!;
    expect(
      acpLaunchFromCommandText(claude.default_command.join(" "), ACP_PROVIDERS),
    ).toEqual({ server: "claude-code", command: [], args: [] });
  });

  it("parses a custom command into its argv", () => {
    expect(
      acpLaunchFromCommandText(
        "my-acp-wrapper --flag 'two words'",
        ACP_PROVIDERS,
      ),
    ).toEqual({
      server: "custom",
      command: ["my-acp-wrapper", "--flag", "two words"],
      args: [],
    });
  });

  it("does not launch an empty custom command", () => {
    expect(acpLaunchFromCommandText("", ACP_PROVIDERS)).toBeNull();
    expect(acpLaunchFromCommandText("   ", ACP_PROVIDERS)).toBeNull();
  });
});

describe("isAcpLaunchDiscoverable", () => {
  it("asks a built-in provider even when no command is stored", () => {
    expect(
      isAcpLaunchDiscoverable({
        server: "claude-code",
        command: [],
        args: [],
      }),
    ).toBe(true);
  });

  it("asks a custom launch that has a command", () => {
    expect(
      isAcpLaunchDiscoverable({
        server: "custom",
        command: ["my-acp-wrapper"],
        args: [],
      }),
    ).toBe(true);
  });

  it("skips a custom launch without a command", () => {
    expect(
      isAcpLaunchDiscoverable({ server: "custom", command: [], args: [] }),
    ).toBe(false);
    expect(isAcpLaunchDiscoverable(null)).toBe(false);
  });

  it("skips a server Canvas does not know when no command is stored", () => {
    expect(
      isAcpLaunchDiscoverable({
        server: "some-future-acp",
        command: [],
        args: [],
      }),
    ).toBe(false);
  });
});

describe("acpLaunchArgv", () => {
  it("joins command and args into the launch identity", () => {
    expect(
      acpLaunchArgv({
        server: "custom",
        command: ["my-acp-wrapper", "--flag"],
        args: ["--extra"],
      }),
    ).toEqual(["my-acp-wrapper", "--flag", "--extra"]);
  });
});
