/**
 * The only module that knows a setup entry configures an *automation*.
 *
 * The published contract states just what varies between entries; everything
 * derivable is the host's to generate. That derivation is here, and nowhere
 * else, so the rest of `src/manifests/` stays about forms rather than about
 * automations.
 *
 * The algorithm mirrors `tests/test_automation_setup.py::_render_payload` in
 * `OpenHands/extensions`, which is the authoritative reference: it produces the
 * request bodies published in that repository's contract fixtures, and those
 * were verified against the live service. The create model is `extra="forbid"`,
 * so any divergence is a hard 422 rather than a dropped field.
 */

import { findAutomationCommand } from "#/utils/automation-catalog";
import { getAutomationEndpoint } from "./automation-interface";
import { collectFields } from "./manifest-local-validation";
import { interpolateText } from "./manifest-template";
import type {
  SetupBlock,
  SetupEntry,
  SetupFormValues,
  SetupRequestBody,
  SetupTriggerKind,
} from "./types";

/** The creation endpoint a derived draft would be posted to. */
export const AUTOMATION_CREATE_ENDPOINT = getAutomationEndpoint("createPrompt");

/**
 * Trigger properties a form field may fill, per trigger kind. A field under a
 * trigger kind whose name is listed here fills the trigger property of the same
 * name; anything else there, such as a phrase to match, is an input to `filter`.
 */
const TRIGGER_PROPERTIES: Record<SetupTriggerKind, readonly string[]> = {
  cron: ["schedule", "timezone"],
  event: ["on"],
};

/**
 * Repository properties a form field may fill, read the same way
 * {@link TRIGGER_PROPERTIES} is: a field whose name is listed here fills the
 * `repos[]` property of the same name. `url` and `provider` are absent because
 * neither is answered by a field of its own — they come from the repo-picker's
 * value and from its declared provider.
 *
 * These two lists are the only field names the derivation knows; everything
 * else it reads it finds by field type. An entry that wants a base branch in
 * its request therefore has to name that field `ref`.
 */
const REPO_PROPERTIES: readonly string[] = ["ref"];

const FORM_PLACEHOLDER_PATTERN = /\{\{form\.([A-Za-z0-9_.]+)\}\}/g;

/** The repository input, which supplies `repos` and an event trigger's source. */
function findRepoPickerField(setup: SetupBlock) {
  const match = Object.entries(collectFields(setup)).find(
    ([, field]) => field.type === "repo-picker",
  );
  return match ? { name: match[0], field: match[1] } : null;
}

/** The single trigger kind a direct entry declares, with its fields. */
function getTrigger(setup: SetupBlock) {
  const entries = Object.entries(setup.form.triggers ?? {});
  if (entries.length === 0) return null;
  const [kind, fields] = entries[0];
  return { kind: kind as SetupTriggerKind, fields: fields ?? {} };
}

/**
 * The create request body these form values produce.
 *
 * No entry declares it: `name` comes from the entry, `repos` from the
 * repo-picker field and its declared provider, and `trigger` from the key and
 * fields under `form.triggers`. Only `prompt` and an event `filter` are
 * declared, because only they cannot be read off the form.
 *
 * Returns null for an assisted entry, which hands setup to a conversation
 * instead of sending a body.
 */
export function buildCreatePayload(
  entry: SetupEntry,
  values: SetupFormValues,
): SetupRequestBody | null {
  const { setup } = entry;
  if (setup.mode !== "direct" || !setup.prompt) return null;

  const scope = { form: values, automation: entry };
  const repoPicker = findRepoPickerField(setup);
  const repository = repoPicker ? values[repoPicker.name] : undefined;

  const payload: SetupRequestBody = {
    name: repository ? `${entry.name} - ${repository}` : entry.name,
    prompt: interpolateText(setup.prompt, scope),
  };

  if (repository && repoPicker?.field.provider) {
    const declared = REPO_PROPERTIES.filter((name) => name in values);
    payload.repos = [
      {
        url: repository,
        ...Object.fromEntries(declared.map((name) => [name, values[name]])),
        provider: repoPicker.field.provider,
      },
    ];
  }

  const trigger = getTrigger(setup);
  if (trigger) {
    const properties = TRIGGER_PROPERTIES[trigger.kind];
    const declared = Object.keys(trigger.fields).filter((name) =>
      properties.includes(name),
    );

    payload.trigger = {
      type: trigger.kind,
      ...Object.fromEntries(declared.map((name) => [name, values[name] ?? ""])),
      ...(trigger.kind === "event" && {
        source: repoPicker?.field.provider ?? "",
        // A filter is optional: an entry that declares none accepts every
        // delivered event, so the key is left off rather than sent empty.
        ...(setup.filter && { filter: interpolateText(setup.filter, scope) }),
      }),
    };
  }

  return payload;
}

/**
 * The preflight body the host sends. The same shape for every entry, so no
 * entry declares it.
 */
export function buildPreflightBody(
  entry: SetupEntry,
  values: SetupFormValues,
): SetupRequestBody | null {
  const draft = buildCreatePayload(entry, values);
  if (!draft) return null;

  return {
    automationId: entry.id,
    endpoint: AUTOMATION_CREATE_ENDPOINT,
    draft,
  };
}

/**
 * What an assisted entry sends into the conversation that finishes setup.
 *
 * The command is not declared either: it lives once, in the owning skill's own
 * `triggers` frontmatter, and the skill defaults to the entry's id. An entry
 * whose skill is invoked by description rather than by command contributes
 * nothing here, and the answers alone open the conversation.
 */
export function buildAssistedMessage(
  entry: SetupEntry,
  values: SetupFormValues,
): string {
  const command = findAutomationCommand(entry);
  const message = entry.setup.message
    ? interpolateText(entry.setup.message, { form: values, automation: entry })
    : "";

  return [command, message].filter(Boolean).join("\n\n");
}

function collectPlaceholderFields(value: string): string[] {
  const names = Array.from(
    value.matchAll(FORM_PLACEHOLDER_PATTERN),
    (match) => match[1],
  );
  return Array.from(new Set(names));
}

/**
 * Which form fields built each payload path.
 *
 * Preflight and the create endpoint reject a draft by payload path, and the
 * host has to turn that back into a highlighted input. Building the body with
 * each field standing in for its own value recovers the mapping exactly, so an
 * entry does not declare it.
 */
export function deriveErrorMap(entry: SetupEntry): Record<string, string[]> {
  const mapping: Record<string, string[]> = {};

  const standIns = Object.fromEntries(
    Object.keys(collectFields(entry.setup)).map((name) => [
      name,
      `{{form.${name}}}`,
    ]),
  );
  const template = buildCreatePayload(entry, standIns);
  if (!template) return mapping;

  const walk = (node: unknown, path: string): void => {
    if (Array.isArray(node)) {
      node.forEach((item, index) => walk(item, `${path}[${index}]`));
      return;
    }
    if (typeof node === "object" && node !== null) {
      Object.entries(node).forEach(([key, item]) =>
        walk(item, path ? `${path}.${key}` : key),
      );
      return;
    }
    if (typeof node === "string") {
      const names = collectPlaceholderFields(node);
      if (names.length > 0) mapping[path] = names;
    }
  };

  walk(template, "");
  return mapping;
}
