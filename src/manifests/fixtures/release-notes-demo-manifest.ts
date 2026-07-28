import type { ExtensionManifest } from "../types";

/**
 * Neutrality fixture: a manifest that has nothing to do with automations.
 *
 * The host is only allowed to be a generic renderer, so it has to be provable
 * that nothing about it assumes one particular feature. This manifest mounts a
 * route, renders a form, interpolates a review, starts a conversation, and
 * emits analytics — the full host path — without touching an automation
 * identifier, schema, endpoint, or analytics stage name.
 *
 * Registered in development builds only; see `manifest-sources.ts`.
 */
export const RELEASE_NOTES_DEMO_MANIFEST: ExtensionManifest = {
  manifestVersion: "1.0",
  id: "release-notes-draft",
  name: "Draft release notes",
  category: "Writing",
  description:
    "Collect the range and audience for a release, then hand them to an agent that drafts the notes.",
  setupMode: "assisted",

  routes: [{ path: "/ext/demo/release-notes-draft", page: "setup" }],

  requires: {
    integrations: [
      {
        id: "github",
        reason: "Used to read the commits and pull requests in the range.",
        enforcement: "warn",
      },
    ],
    secrets: [],
    onUnmet: {
      behavior: "block",
      message: "Connect the required accounts before continuing.",
    },
    onWarn: {
      behavior: "continue",
      message:
        "GitHub is not connected yet. The agent will ask for the commit range instead of reading it.",
    },
  },

  form: {
    note: "These answers start the conversation. Anything left blank is something the agent will ask about.",
    fields: [
      {
        name: "repository",
        type: "repo-picker",
        label: "Repository",
        help: "The repository the release is cut from.",
        provider: "github",
        required: true,
      },
      {
        name: "fromRef",
        type: "text",
        label: "Previous release",
        help: "Tag or commit the last release was cut at.",
        placeholder: "v1.4.0",
        required: true,
        constraints: { minLength: 1, maxLength: 100 },
      },
      {
        name: "toRef",
        type: "text",
        label: "This release",
        help: "Tag or commit the new release is cut at.",
        default: "main",
        required: true,
        constraints: { minLength: 1, maxLength: 100 },
      },
      {
        name: "audience",
        type: "select",
        label: "Audience",
        help: "How much detail the notes should carry.",
        default: "users",
        required: true,
        options: [
          { value: "users", label: "End users" },
          { value: "developers", label: "Developers" },
        ],
      },
      {
        name: "highlights",
        type: "textarea",
        label: "Anything to lead with?",
        help: "Changes that deserve the top of the notes.",
        required: false,
        constraints: { maxLength: 2000 },
      },
    ],
  },

  validation: {
    onInvalid: { behavior: "blockSubmit", errorTarget: "field" },
  },

  review: {
    title: "Start the drafting conversation",
    note: "Drafting happens in a conversation with an agent. You can change any of this while you talk to it.",
    emptyValueText: "the agent will ask",
    summary: [
      { label: "Repository", value: "{{form.repository}}" },
      { label: "Range", value: "{{form.fromRef}} to {{form.toRef}}" },
      { label: "Audience", value: "{{form.audience}}" },
      { label: "Lead with", value: "{{form.highlights}}" },
    ],
    confirmLabel: "Start drafting",
  },

  submit: {
    action: "conversation.start",
    message:
      "Draft release notes for {{form.repository}}, covering {{form.fromRef}} to {{form.toRef}}. Write them for {{form.audience}}. Lead with: {{form.highlights}}",
    onSuccess: {
      behavior: "navigate",
      to: "/conversations/{{response.conversation_id}}",
    },
    onError: {
      behavior: "stayOnForm",
      errorTarget: "form",
      message: "The drafting conversation could not be started.",
    },
  },

  analytics: {
    consent: "required",
    stages: [
      {
        id: "demo_setup_opened",
        on: "route.entered",
        properties: { manifest_id: "{{manifest.id}}", setup_mode: "assisted" },
      },
      {
        id: "demo_setup_handed_off",
        on: "submit.succeeded",
        properties: { manifest_id: "{{manifest.id}}", setup_mode: "assisted" },
      },
      {
        id: "demo_setup_failed",
        on: "submit.failed",
        properties: { manifest_id: "{{manifest.id}}", setup_mode: "assisted" },
      },
    ],
  },
};
