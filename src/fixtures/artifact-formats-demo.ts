/**
 * Binary artifact fixtures for mock mode — a real PNG and PDF, so the inline
 * image and PDF previews can be demonstrated and screenshotted without an LLM
 * or a live workspace.
 */
import type {
  ActionEvent,
  ObservationEvent,
  OpenHandsEvent,
} from "#/types/agent-server/core";
import { SecurityRisk } from "#/types/agent-server/core";
import type { FileEditorAction } from "#/types/agent-server/core/base/action";
import type { FileEditorObservation } from "#/types/agent-server/core/base/observation";

export const ARTIFACT_DEMO_PATHS = {
  png: "assets/preview.png",
  pdf: "docs/spec.pdf",
} as const;

export const ARTIFACT_DEMO_MIME: Record<string, string> = {
  [ARTIFACT_DEMO_PATHS.png]: "image/png",
  [ARTIFACT_DEMO_PATHS.pdf]: "application/pdf",
};

/** Base64 payload for a demo artifact path, decoded by the mock fileserver. */
export const ARTIFACT_DEMO_BYTES: Record<string, string> = {
  [ARTIFACT_DEMO_PATHS.png]:
    "iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAIAAABt+uBvAAAAtElEQVR42u3ZsQkAIAxFQTdxRAe3F7FyhNikUA+sA173+KW2Hr4x" +
    "V/hevVMAAQIECBAgQIAAAQIEKAHo58+f3AEECBAgQIAAAQIECBCgDCCxquYBAQIECBAgQIAAAQJkF1PzgAABAgQIECBAgAABsouJ" +
    "VUCAAAECBAgQIECAAAGyi6l5QIAAAQIECBAgQIAA2cXUPCBAgAABAgQIECBAgADZxdQ8IECAAAECBAgQIECALrmzAXxR1pM0AboQ" +
    "AAAAAElFTkSuQmCC",
  [ARTIFACT_DEMO_PATHS.pdf]:
    "JVBERi0xLjQKMSAwIG9iajw8L1R5cGUvQ2F0YWxvZy9QYWdlcyAyIDAgUj4+ZW5kb2JqCjIgMCBvYmo8PC9UeXBlL1BhZ2VzL0tp" +
    "ZHNbMyAwIFJdL0NvdW50IDE+PmVuZG9iagozIDAgb2JqPDwvVHlwZS9QYWdlL1BhcmVudCAyIDAgUi9NZWRpYUJveFswIDAgMzAw" +
    "IDE2MF0vQ29udGVudHMgNCAwIFIvUmVzb3VyY2VzPDwvRm9udDw8L0YxIDUgMCBSPj4+Pj4+ZW5kb2JqCjQgMCBvYmo8PC9MZW5n" +
    "dGggOTg+PnN0cmVhbQpCVCAvRjEgMTggVGYgMjQgMTA4IFRkIChBcnRpZmFjdCBwcmV2aWV3KSBUaiBFVApCVCAvRjEgMTEgVGYg" +
    "MjQgODIgVGQgKFBERiByZW5kZXJzIGlubGluZS4pIFRqIEVUCmVuZHN0cmVhbWVuZG9iago1IDAgb2JqPDwvVHlwZS9Gb250L1N1" +
    "YnR5cGUvVHlwZTEvQmFzZUZvbnQvSGVsdmV0aWNhPj5lbmRvYmoKeHJlZgowIDYKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAw" +
    "MDA5IDAwMDAwIG4gCjAwMDAwMDAwNTIgMDAwMDAgbiAKMDAwMDAwMDEwMSAwMDAwMCBuIAowMDAwMDAwMjExIDAwMDAwIG4gCjAw" +
    "MDAwMDAzNTMgMDAwMDAgbiAKdHJhaWxlcjw8L1NpemUgNi9Sb290IDEgMCBSPj4Kc3RhcnR4cmVmCjQxNAolJUVPRgo=",
};

interface ArtifactSpec {
  path: string;
  id: string;
  /** `create` for a fresh artifact, `view` for one the agent read back. */
  command?: "create" | "view";
  content: string;
}

/**
 * Builds the action/observation pair for a file-editor event so the mock
 * conversation renders exactly what the card would in a live session.
 */
function artifactEvents(
  spec: ArtifactSpec,
  offsetSeconds: number,
  timestamp: (offset: number) => string,
): OpenHandsEvent[] {
  const command = spec.command ?? "create";
  const action: ActionEvent<FileEditorAction> = {
    id: `${spec.id}-action`,
    timestamp: timestamp(offsetSeconds),
    source: "agent",
    thought: [],
    thinking_blocks: [],
    action: {
      kind: "FileEditorAction",
      command,
      path: spec.path,
      // Binary artifacts carry no text payload; the preview reads the bytes
      // from the workspace URL instead.
      file_text: command === "create" ? spec.content : null,
      old_str: null,
      new_str: null,
      insert_line: null,
      view_range: null,
    },
    tool_name: "file_editor",
    tool_call_id: `${spec.id}-tool-call`,
    tool_call: {
      id: `${spec.id}-tool-call`,
      type: "function",
      function: {
        name: "file_editor",
        arguments: JSON.stringify({ command, path: spec.path }),
      },
    },
    llm_response_id: `${spec.id}-response`,
    security_risk: SecurityRisk.LOW,
  };

  const observation: ObservationEvent<FileEditorObservation> = {
    id: `${spec.id}-observation`,
    timestamp: timestamp(offsetSeconds + 1),
    source: "environment",
    tool_name: "file_editor",
    tool_call_id: `${spec.id}-tool-call`,
    action_id: action.id,
    observation: {
      kind: "FileEditorObservation",
      command,
      output: `Created ${spec.path}`,
      path: spec.path,
      prev_exist: false,
      old_content: null,
      new_content: command === "create" ? spec.content : null,
      error: null,
    },
  };

  return [action, observation];
}

/** One action/observation pair per binary artifact, in a fixed order. */
export function createArtifactFormatEvents(
  startOffsetSeconds: number,
  timestamp: (offset: number) => string,
): OpenHandsEvent[] {
  const specs: ArtifactSpec[] = [
    { path: ARTIFACT_DEMO_PATHS.png, id: "canvas-demo-png", content: "" },
    { path: ARTIFACT_DEMO_PATHS.pdf, id: "canvas-demo-pdf", content: "" },
  ];

  return specs.flatMap((spec, index) =>
    artifactEvents(spec, startOffsetSeconds + index * 2, timestamp),
  );
}
