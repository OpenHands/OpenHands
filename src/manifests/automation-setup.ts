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

import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { findAutomationCommand } from "#/utils/automation-catalog";
import { getAutomationEndpoint } from "./automation-interface";
import {
  collectFields,
  fieldText,
  fieldValues,
} from "./manifest-local-validation";
import { interpolateText, interpolateValue } from "./manifest-template";
import type {
  SetupBlock,
  SetupBundleConfigValue,
  SetupEntry,
  SetupFormValues,
  SetupRequestBody,
  SetupTriggerKind,
} from "./types";

/**
 * The creation endpoint a derived draft would be posted to. Resolved on call
 * rather than at import, because the endpoint is the interface manifest's and
 * this module loads whether or not one was admitted.
 *
 * A bundle entry is created through the raw endpoint, because what it sends is
 * a tarball it uploaded rather than arguments to a preset. Called without an
 * entry - as the import path does - it answers for a prompt.
 */
export function automationCreateEndpoint(entry?: SetupEntry): string {
  if (entry && isBundleEntry(entry)) {
    return requireBundleEndpoint("createBundle");
  }
  return getAutomationEndpoint("createPrompt");
}

/** Where a bundle's tarball is uploaded, before the create call. */
export function automationUploadEndpoint(): string {
  return requireBundleEndpoint("uploads");
}

/** The endpoints a bundle entry cannot be created without. */
const BUNDLE_ENDPOINTS = ["createBundle", "uploads"] as const;

/**
 * An endpoint only a bundle needs. The interface manifest may predate bundles,
 * and the host holds no path of its own to fall back to, so this is where that
 * runs out rather than somewhere deep in a request.
 */
function requireBundleEndpoint(
  name: (typeof BUNDLE_ENDPOINTS)[number],
): string {
  const path = getAutomationEndpoint(name);
  if (!path) {
    throw new Error(
      `The published automation interface declares no '${name}' endpoint, ` +
        "so this deployment cannot create an automation from a script bundle.",
    );
  }
  return path;
}

/**
 * The endpoints this entry needs that the published interface does not declare.
 *
 * Asked before the form renders rather than discovered at the moment of
 * creating: a pinned package that predates bundles can never answer, and the
 * dialog can say so while nothing has been filled in yet. Empty for every
 * entry that is not a bundle, which needs nothing beyond what the block has
 * always declared.
 */
export function missingCreateEndpoints(entry: SetupEntry): string[] {
  if (!isBundleEntry(entry)) return [];
  return BUNDLE_ENDPOINTS.filter((name) => !getAutomationEndpoint(name));
}

/** Whether this entry ships a script tarball instead of a prompt. */
export function isBundleEntry(entry: SetupEntry): boolean {
  return entry.setup.mode === "direct" && entry.setup.bundle !== undefined;
}

/**
 * The `tarball_path` a preflight draft carries.
 *
 * Preflight runs on every field blur and the upload happens once, at submit,
 * so there is no real path to send yet. The service checks this field's scheme
 * at preflight and its ownership only at creation, so a well-formed stand-in
 * validates exactly what preflight is for - the rest of the body - without
 * uploading an archive per keystroke.
 */
export const PREFLIGHT_TARBALL_PATH =
  "oh-internal://uploads/00000000-0000-0000-0000-000000000000";

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

/** Every repository the form collected, whether the picker takes one or many. */
function repositories(setup: SetupBlock, values: SetupFormValues): string[] {
  const picker = findRepoPickerField(setup);
  return picker ? fieldValues(values[picker.name]) : [];
}

/**
 * The created automation's name.
 *
 * One repository is worth naming; several are not, so the count stands in
 * rather than a list of names that would not fit.
 */
function deriveName(entry: SetupEntry, values: SetupFormValues): string {
  const repos = repositories(entry.setup, values);
  if (repos.length === 0) return entry.name;
  if (repos.length === 1) return `${entry.name} - ${repos[0]}`;
  // The count is the one word here the host writes rather than reads off the
  // entry, so it is translated. There is no translator to pass in: the
  // derivation runs from a memo, an upload and a test alike.
  const count = i18n.t(I18nKey.SETUP$REPOSITORY_COUNT, {
    total: repos.length,
  });
  return `${entry.name} - ${count}`;
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
  /** Bundle entries only: what the upload returned. */
  tarballPath: string = PREFLIGHT_TARBALL_PATH,
): SetupRequestBody | null {
  const { setup } = entry;
  if (setup.mode !== "direct") return null;
  if (setup.bundle) return buildBundlePayload(entry, values, tarballPath);
  if (!setup.prompt) return null;

  const scope = { form: values, automation: entry };
  const repoPicker = findRepoPickerField(setup);
  const repos = repositories(setup, values);

  const payload: SetupRequestBody = {
    name: deriveName(entry, values),
    prompt: interpolateText(setup.prompt, scope),
  };

  if (repos.length > 0 && repoPicker?.field.provider) {
    const declared = REPO_PROPERTIES.filter((name) => name in values);
    payload.repos = repos.map((url) => ({
      url,
      ...Object.fromEntries(
        declared.map((name) => [name, fieldText(values[name])]),
      ),
      provider: repoPicker.field.provider as string,
    }));
  }

  // A filter is optional: an entry that declares none accepts every delivered
  // event, so the key is left off rather than sent empty.
  const trigger = buildTrigger(entry, values);
  if (trigger) payload.trigger = trigger;

  const template = buildTemplate(entry, values);
  if (template) payload.template = template;

  return payload;
}

/**
 * The `template` provenance a prompt entry sends, or undefined for one that
 * publishes no version.
 *
 * It is what lets the service recognise a second setup of the same entry as
 * the automation it already holds. The bundle path derives the same block from
 * the config the bundle declares; a prompt entry declares none, so the answer
 * to "what was this created from" is the form as submitted.
 */
function buildTemplate(
  entry: SetupEntry,
  values: SetupFormValues,
): SetupRequestBody | undefined {
  if (!entry.version) return undefined;
  return {
    id: entry.id,
    version: entry.version,
    config: { ...values },
  };
}

/**
 * The `trigger` object, read off the key and fields under `form.triggers`.
 *
 * Identical for both kinds of direct entry: only the create endpoint and what
 * the automation is told to do differ between a prompt and a bundle.
 */
function buildTrigger(
  entry: SetupEntry,
  values: SetupFormValues,
): SetupRequestBody | undefined {
  const trigger = getTrigger(entry.setup);
  if (!trigger) return undefined;

  const properties = TRIGGER_PROPERTIES[trigger.kind];
  const declared = Object.keys(trigger.fields).filter((name) =>
    properties.includes(name),
  );
  const repoPicker = findRepoPickerField(entry.setup);

  return {
    type: trigger.kind,
    ...Object.fromEntries(
      declared.map((name) => [name, fieldText(values[name])]),
    ),
    ...(trigger.kind === "event" && {
      source: repoPicker?.field.provider ?? "",
      ...(entry.setup.filter && {
        filter: interpolateText(entry.setup.filter, {
          form: values,
          automation: entry,
        }),
      }),
    }),
  };
}

/**
 * The raw create body a bundle entry produces.
 *
 * `tarball_path` is the one value neither declared nor derived: the host packs
 * and uploads the bundle first, and creates from what came back. `template` is
 * the provenance that makes enabling the same entry twice return the
 * automation that already exists rather than a second one.
 *
 * There is no `repos`: the raw endpoint has no such field, and a bundle's
 * script fetches what it needs itself.
 */
function buildBundlePayload(
  entry: SetupEntry,
  values: SetupFormValues,
  tarballPath: string,
): SetupRequestBody {
  const bundle = entry.setup.bundle!;
  const scope = { form: values, automation: entry };

  const payload: SetupRequestBody = {
    name: deriveName(entry, values),
  };

  const trigger = buildTrigger(entry, values);
  if (trigger) payload.trigger = trigger;

  payload.tarball_path = tarballPath;
  payload.entrypoint = bundle.entrypoint;
  if (bundle.setupScript) payload.setup_script_path = bundle.setupScript;
  if (bundle.timeout !== undefined) payload.timeout = bundle.timeout;
  payload.template = {
    id: entry.id,
    version: bundle.version,
    config: interpolateConfig(bundle.config, scope) as SetupRequestBody,
  };

  return payload;
}

/**
 * Placeholder substitution over the config tree. Only string leaves are
 * templated; a number, a boolean or a null is written through as itself, so an
 * entry can state a value the script reads as the type it expects.
 */
function interpolateConfig(
  node: SetupBundleConfigValue,
  scope: Parameters<typeof interpolateText>[1],
): SetupBundleConfigValue {
  if (typeof node === "string") {
    return interpolateValue(node, scope);
  }
  if (Array.isArray(node)) {
    return node.map((item) => interpolateConfig(item, scope));
  }
  if (typeof node === "object" && node !== null) {
    return Object.fromEntries(
      Object.entries(node).map(([key, value]) => [
        key,
        interpolateConfig(value, scope),
      ]),
    );
  }
  return node;
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
    endpoint: automationCreateEndpoint(entry),
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
