import { describe, expect, it } from "vitest";

import {
  AUTOMATION_FORM_UPDATE_ACTION_KIND,
  AUTOMATION_FORM_UPDATE_CLIENT_TOOL,
  AUTOMATION_FORM_UPDATE_TOOL_NAME,
} from "#/api/automation-form-client-tool";

describe("automation_form_update client tool", () => {
  it("exports the semantic tool name and generated action kind", () => {
    expect(AUTOMATION_FORM_UPDATE_TOOL_NAME).toBe("automation_form_update");
    expect(AUTOMATION_FORM_UPDATE_ACTION_KIND).toBe(
      "ClientAction_automation_form_update",
    );
    expect(AUTOMATION_FORM_UPDATE_CLIENT_TOOL.name).toBe(
      AUTOMATION_FORM_UPDATE_TOOL_NAME,
    );
  });

  it("pins the validated form patch schema", () => {
    expect(AUTOMATION_FORM_UPDATE_CLIENT_TOOL.parameters).toMatchObject({
      type: "object",
      additionalProperties: false,
      properties: {
        fields: {
          type: "object",
          additionalProperties: false,
          properties: {
            kind: { enum: ["prompt", "plugin", "custom"] },
            triggerKind: { enum: ["cron", "event"] },
            frequency: {
              enum: ["once", "hourly", "daily", "weekdays", "weekly", "custom"],
            },
            customCode: { type: "string" },
            setupScript: { type: "string" },
          },
        },
        overwrite_user_edits: { type: "boolean" },
      },
      required: ["fields"],
    });
  });

  it("is annotated as a local UI update tool", () => {
    expect(AUTOMATION_FORM_UPDATE_CLIENT_TOOL.annotations).toEqual({
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: false,
    });
  });
});
