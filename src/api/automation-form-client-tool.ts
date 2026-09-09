import { AUTOMATION_FORM_UPDATE_TOOL_NAME } from "#/constants/automation-form";
import type { ClientToolSpec } from "./canvas-ui-client-tool";

export {
  AUTOMATION_FORM_UPDATE_ACTION_KIND,
  AUTOMATION_FORM_UPDATE_TOOL_NAME,
} from "#/constants/automation-form";

const AUTOMATION_FORM_UPDATE_DESCRIPTION = `Update the automation setup form that is open beside this conversation. Use this only when the user asks you to help fill, revise, or complete automation configuration fields in the UI.

The user can keep editing the form while you converse. By default, your updates fill empty fields or fields the user has not manually edited. If the user already edited a field, do not overwrite it unless they explicitly asked you to replace their value; set overwrite_user_edits=true only in that case.

Useful fields:
* kind: "prompt", "plugin", or "custom".
* name: short automation name.
* prompt: natural-language instruction for prompt/plugin automations.
* repository: optional GitHub repository URL or owner/repo for prompt/plugin automations.
* pluginSource and pluginRef: plugin source and optional ref for plugin automations.
* customCode, entrypoint, setupScriptPath, setupScript: custom Python bundle fields.
* triggerKind: "cron" or "event".
* frequency: "once", "hourly", "daily", "weekdays", "weekly", or "custom" for cron triggers.
* time, timezone, customSchedule: cron scheduling fields.
* eventSource, eventKey, eventFilter: event trigger fields.
* showTimeout and timeoutSeconds: optional timeout controls.

Call this with only the fields you are changing. After calling it, briefly tell the user which fields you filled and ask them to review anything uncertain.`;

export const AUTOMATION_FORM_UPDATE_CLIENT_TOOL: ClientToolSpec = {
  name: AUTOMATION_FORM_UPDATE_TOOL_NAME,
  description: AUTOMATION_FORM_UPDATE_DESCRIPTION,
  parameters: {
    type: "object",
    additionalProperties: false,
    properties: {
      fields: {
        type: "object",
        additionalProperties: false,
        properties: {
          kind: { type: "string", enum: ["prompt", "plugin", "custom"] },
          name: { type: "string" },
          prompt: { type: "string" },
          repository: { type: "string" },
          pluginSource: { type: "string" },
          pluginRef: { type: "string" },
          customCode: { type: "string" },
          entrypoint: { type: "string" },
          setupScriptPath: { type: "string" },
          setupScript: { type: "string" },
          triggerKind: { type: "string", enum: ["cron", "event"] },
          frequency: {
            type: "string",
            enum: ["once", "hourly", "daily", "weekdays", "weekly", "custom"],
          },
          time: { type: "string", description: "24-hour HH:MM time." },
          timezone: { type: "string" },
          customSchedule: { type: "string", description: "Cron expression." },
          eventSource: { type: "string" },
          eventKey: { type: "string" },
          eventFilter: { type: "string" },
          showTimeout: { type: "boolean" },
          timeoutSeconds: { type: "string" },
        },
        description: "Automation form field values to patch in the visible UI.",
      },
      overwrite_user_edits: {
        type: "boolean",
        description:
          "Set true only when the user explicitly asks you to replace values they edited manually.",
      },
    },
    required: ["fields"],
  },
  annotations: {
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: false,
    openWorldHint: false,
  },
};
