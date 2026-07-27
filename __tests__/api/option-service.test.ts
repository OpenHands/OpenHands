import { beforeEach, describe, expect, it, vi } from "vitest";
import { http, HttpResponse } from "msw";
import {
  AgentServerUnavailableError,
  AgentServerUnknownVersionError,
  AgentServerUnsupportedVersionError,
  clearCachedAgentServerInfo,
  isAgentServerToolAvailable,
  MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
} from "#/api/agent-server-compatibility";
import OptionService from "#/api/option-service/option-service.api";
import { server } from "#/mocks/node";

describe("OptionService", () => {
  beforeEach(() => {
    clearCachedAgentServerInfo();
  });

  it("returns config in mock mode without a live backend", async () => {
    const config = await OptionService.getConfig();

    expect(config.feature_flags.hide_llm_settings).toBe(false);
    expect(config.feature_flags.hide_users_page).toBe(true);
    expect(config.updated_at).toBeTruthy();
  });

  it("loads config when the agent server is compatible", async () => {
    server.use(
      http.get("*/server_info", () =>
        HttpResponse.json({
          uptime: 0,
          idle_time: 0,
          version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
        }),
      ),
    );

    await expect(OptionService.getConfig()).resolves.toMatchObject({
      feature_flags: expect.objectContaining({ hide_llm_settings: false }),
    });
  });

  it("throws when the agent server is too old", async () => {
    server.use(
      http.get("*/server_info", () =>
        HttpResponse.json({ uptime: 0, idle_time: 0, version: "1.27.1" }),
      ),
    );

    await expect(OptionService.getConfig()).rejects.toMatchObject({
      name: AgentServerUnsupportedVersionError.name,
      actualVersion: "1.27.1",
      requiredVersion: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });
  });

  it("throws when the server does not advertise a version", async () => {
    server.use(
      http.get("*/server_info", () =>
        HttpResponse.json({ uptime: 0, idle_time: 0 }),
      ),
    );

    await expect(OptionService.getConfig()).rejects.toMatchObject({
      name: AgentServerUnknownVersionError.name,
      actualVersion: null,
    });
  });

  it("throws an unavailable error when the agent server cannot be reached", async () => {
    server.use(http.get("*/server_info", () => HttpResponse.error()));

    await expect(OptionService.getConfig()).rejects.toMatchObject({
      name: AgentServerUnavailableError.name,
      message: expect.stringContaining(
        "Could not connect to the configured agent server",
      ),
      details: expect.stringContaining("Request failed"),
    });
  });

  it("caches usable_tools from server_info for later tool gating", async () => {
    server.use(
      http.get("*/server_info", () =>
        HttpResponse.json({
          uptime: 0,
          idle_time: 0,
          version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
          usable_tools: ["terminal", "file_editor", "task_tracker"],
        }),
      ),
    );

    await OptionService.getConfig();

    expect(isAgentServerToolAvailable("browser_tool_set")).toBe(false);
    expect(isAgentServerToolAvailable("terminal")).toBe(true);
  });

  it("allows all tools when the server does not advertise tool metadata", async () => {
    server.use(
      http.get("*/server_info", () =>
        HttpResponse.json({
          uptime: 0,
          idle_time: 0,
          version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
        }),
      ),
    );

    await OptionService.getConfig();

    expect(isAgentServerToolAvailable("browser_tool_set")).toBe(true);
    expect(isAgentServerToolAvailable("terminal")).toBe(true);
  });

  it("returns models from mocked LLM endpoints", async () => {
    const models = await OptionService.getModels();

    expect(models.models).toContain("openhands/claude-opus-4-5-20251101");
    expect(models.models).toContain("openai/gpt-5.5");
    expect(models.verified_models).toContain("claude-opus-4-5-20251101");
    expect(models.verified_models).toContain("gpt-5.5");
    expect(models.verified_providers).toEqual([
      "anthropic",
      "openai",
      "openhands",
    ]);
    expect(models.default_model).toBeTruthy();
  });
});
