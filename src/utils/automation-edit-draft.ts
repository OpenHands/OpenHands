/**
 * Turn a saved automation into the setup-page draft the editor opens.
 *
 * Edit reuses the create form, so the draft has to carry the same fields
 * the panel already knows how to render. `editingAutomationId` is what
 * switches that page from "save draft / create" to "save / test".
 */
import {
  type AutomationSetupDraft,
  type AutomationSetupFormPatch,
  type AutomationSetupFrequency,
  type AutomationSetupKind,
} from "#/api/automation-setup-types";
import type { Automation } from "#/types/automation";
import {
  formatTimeOfDay,
  parseCronSchedule,
} from "#/utils/automation-schedule";

const HOURLY_CRON = "0 * * * *";

function setupKind(automation: Automation): AutomationSetupKind {
  if ((automation.plugins?.length ?? 0) > 0) return "plugin";
  if (!automation.prompt?.trim() && automation.entrypoint) return "custom";
  return "prompt";
}

function eventKey(on: Automation["trigger"]["on"]): string {
  if (Array.isArray(on)) return on.join(",");
  return on ?? "";
}

/**
 * Map a cron expression onto the setup form's frequency controls.
 *
 * Presets the shared parser already understands (daily, weekdays, weekly)
 * stay presets. Hourly is recognized here because that parser treats
 * `0 * * * *` as custom. Everything else, including one-time dates, stays
 * a raw cron so we do not invent a date the expression does not contain.
 */
function scheduleFields(
  schedule: string | undefined,
  timezone: string | undefined,
): AutomationSetupFormPatch {
  const raw = schedule?.trim() ?? "";
  const zone = timezone?.trim() ?? "";
  if (raw === HOURLY_CRON) {
    return {
      triggerKind: "cron",
      frequency: "hourly",
      ...(zone ? { timezone: zone } : {}),
    };
  }

  const parsed = parseCronSchedule(raw);
  if (parsed.kind === "custom") {
    return {
      triggerKind: "cron",
      frequency: "custom",
      customSchedule: parsed.raw,
      ...(zone ? { timezone: zone } : {}),
    };
  }

  const frequency: AutomationSetupFrequency = parsed.kind;
  return {
    triggerKind: "cron",
    frequency,
    time: formatTimeOfDay(parsed.hour, parsed.minute),
    ...(frequency === "weekly" ? { weekday: String(parsed.weekday ?? 1) } : {}),
    ...(zone ? { timezone: zone } : {}),
  };
}

/** Form values for one existing automation, ready to store on a conversation. */
export function setupDraftFromAutomation(
  automation: Automation,
): AutomationSetupDraft {
  const kind = setupKind(automation);
  const prompt = automation.prompt ?? "";
  const plugins = (automation.plugins ?? []).filter(
    (plugin) => plugin.trim().length > 0,
  );
  const trigger = automation.trigger;
  const form: AutomationSetupFormPatch = {
    kind,
    name: automation.name,
    prompt,
    repository: automation.repository ?? "",
    model: automation.model ?? "",
    agentProfileId: automation.agent_profile_id ?? "",
    entrypoint: automation.entrypoint ?? "",
    setupScriptPath: automation.setup_script_path ?? "",
    ...(trigger.type === "event"
      ? {
          triggerKind: "event" as const,
          eventSource: trigger.source ?? "",
          eventKey: eventKey(trigger.on),
          eventFilter: trigger.filter ?? "",
        }
      : scheduleFields(
          trigger.schedule,
          trigger.timezone ?? automation.timezone,
        )),
  };

  if (automation.timeout != null) {
    form.showTimeout = true;
    form.timeoutSeconds = String(automation.timeout);
  }

  return {
    prompt,
    kind,
    editingAutomationId: automation.id,
    ...(kind === "custom" && automation.tarball_path
      ? { existingCustomTarballPath: automation.tarball_path }
      : {}),
    ...(plugins.length > 0 ? { plugins } : {}),
    form,
  };
}
