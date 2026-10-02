/**
 * Binary artifact fixtures for mock mode — a PNG, a PDF, and a real
 * `.docx` / `.xlsx` / `.pptx` package each, so the extended inline previews can
 * be demonstrated and screenshotted without an LLM or a live workspace.
 *
 * The Office packages are genuine ZIP containers (deflate-raw entries, a
 * central directory, and an end-of-central-directory record), so they exercise
 * the same `readOoxmlPreview` path a real document would.
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
  docx: "docs/plan.docx",
  xlsx: "docs/budget.xlsx",
  pptx: "docs/deck.pptx",
} as const;

export const ARTIFACT_DEMO_MIME: Record<string, string> = {
  [ARTIFACT_DEMO_PATHS.png]: "image/png",
  [ARTIFACT_DEMO_PATHS.pdf]: "application/pdf",
  [ARTIFACT_DEMO_PATHS.docx]:
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  [ARTIFACT_DEMO_PATHS.xlsx]:
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  [ARTIFACT_DEMO_PATHS.pptx]:
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
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
  [ARTIFACT_DEMO_PATHS.docx]:
    "UEsDBBQAAAAIAJRYPl3GEnoHrAAAAPEAAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbF2Puw7CMAxFf6XKihoXBgaUlIEdGPgBK3Hb" +
    "iOahJBT4exKQOjBax/dcWxxfdm4Wisl4J9mWd+zYi9s7UGoKcUmyKedwAEhqIouJ+0CukMFHi7mMcYSA6o4jwa7r9qC8y+Rym6uD" +
    "9eJS5NFoaq4Y8xktSQZPHzVorx62bPJiY83pF6vNkmEIs1GYy02wOP3X2fphMIrWfLWF6BWlZNxoZ74Si8Ztqh56Ad+n+g9QSwME" +
    "FAAAAAgAlFg+XdrzSNAyAQAADAMAABEAAAB3b3JkL2RvY3VtZW50LnhtbLVSzW7CMAy+8xRWH6CBHXaooGjaNG0nEEi7m8ShkfKn" +
    "JKXj7demwKQxNu2wS2LH/n4ceb58NxoOFKJydlHMymmxrOddJRxvDdkEfdnGqlsUTUq+YizyhgzG0nmyfU26YDD1adizzgXhg+MU" +
    "o7J7o9nddHrPDCpbDJQ7J471pA/8kPl1yNc2HTVBVx1QL4oXQtFDZwWr5+zSk49Ub0gTRgKv0Q7VlHvC2PlJfOreNspDaggwJCWR" +
    "J/CBDoo62FGjrAAESZjaQCA17ssfGE82bGsuwas4Wz57PVWvXL+hVgITwUpKxQlcm7SyFP9L79FZqYLJo6+fnmEYmQJoh+KGZtrp" +
    "DB0JeP3lI1edpXCNzC/8e8iDxdsAdpG6Ibih0fPfNDetUL+KsnHayRDkdRyC86rXH1BLAQIUAxQAAAAIAJRYPl3GEnoHrAAAAPEA" +
    "AAATAAAAAAAAAAAAAACAAQAAAABbQ29udGVudF9UeXBlc10ueG1sUEsBAhQDFAAAAAgAlFg+XdrzSNAyAQAADAMAABEAAAAAAAAA" +
    "AAAAAIAB3QAAAHdvcmQvZG9jdW1lbnQueG1sUEsFBgAAAAACAAIAgAAAAD4CAAAAAA==",
  [ARTIFACT_DEMO_PATHS.xlsx]:
    "UEsDBBQAAAAIAJRYPl2VnK/BrwAAACgBAAAPAAAAeGwvd29ya2Jvb2sueG1sjZBLDoJADIavMukBHGDhgvCIxg3HGKHABOaRdlCP" +
    "7wiS4M5VX1/+v21Rv8wsHkisnS0hPSVQV8XT0XR3bhJxaLmEMQSfS8ntiEbxyXm0cdI7MirEkgbJnlB1PCIGM8ssSc7SKG1hU8jp" +
    "Hw3X97rFm2sXgzZsIoSzCnE1HrVnqIrVgb9RWGWwhOvSDRhArL2mi1eAoFzHhJouBflLX9qwqJkPeHbAsw8udxe5P6J6A1BLAwQU" +
    "AAAACACUWD5dISzZuH8AAAC4AAAAFAAAAHhsL3NoYXJlZFN0cmluZ3MueG1sZc4xDsIwDAXQq1Q5QB0YGFCaDkzMnCAC00Sq48i2" +
    "Ko5PKgYkGP/70tcP84vWYUPRwnVyh9G7OQZVG7pXnVw2a2cAvWekpCM3rL15slCyHmUBbYLpoRnRaIWj9yegVKrrMyUGi1dDCmAx" +
    "wJ4/dmG1X7uh7D/+2FjSgl+G/i6+AVBLAwQUAAAACACUWD5dSSktjYAAAAAXAQAAGAAAAHhsL3dvcmtzaGVldHMvc2hlZXQxLnht" +
    "bG3PSwoCMQwG4KuUHMC0qQsXmQyKFxmGguKj0JbOHN86Si3FVR5/+CA8ro+7yi7Eq38OYHYaRuHFh1u8OJeEt3Ke0iQc/KJCuQHh" +
    "+d0cDag0QCxzFs2YhXH+Zqc2MzXDYlSIKkTNMXUQfQjS+r9iq2IbxXaK3bb7Q49g8x/+3n4BUEsDBBQAAAAIAJRYPl3uhOhCcAAA" +
    "ANIAAAAYAAAAeGwvd29ya3NoZWV0cy9zaGVldDIueG1ss7GvyM1RKEstKs7Mz7NVMtQzULK3synPL8ouzkhNLbGzAVMuiSWJdjZF" +
    "+eUKRUA1SnY2ySCGo6GSQomtUjGQX2ZnYKNfZmejnwyVc0KWM4TL6QPNgBtkBDfICEmxEZpBRhAjDE0N0EzRR3KaPsLFAFBLAQIU" +
    "AxQAAAAIAJRYPl2VnK/BrwAAACgBAAAPAAAAAAAAAAAAAACAAQAAAAB4bC93b3JrYm9vay54bWxQSwECFAMUAAAACACUWD5dISzZ" +
    "uH8AAAC4AAAAFAAAAAAAAAAAAAAAgAHcAAAAeGwvc2hhcmVkU3RyaW5ncy54bWxQSwECFAMUAAAACACUWD5dSSktjYAAAAAXAQAA" +
    "GAAAAAAAAAAAAAAAgAGNAQAAeGwvd29ya3NoZWV0cy9zaGVldDEueG1sUEsBAhQDFAAAAAgAlFg+Xe6E6EJwAAAA0gAAABgAAAAA" +
    "AAAAAAAAAIABQwIAAHhsL3dvcmtzaGVldHMvc2hlZXQyLnhtbFBLBQYAAAAABAAEAAsBAADpAgAAAAA=",
  [ARTIFACT_DEMO_PATHS.pptx]:
    "UEsDBBQAAAAIAJRYPl1jzdLrngAAACABAAAUAAAAcHB0L3ByZXNlbnRhdGlvbi54bWyNj0sKwkAMhq9ScoCmLVhh6GPjpuAlhs7U" +
    "DsyLZBSP71RFqit3f/IlH0k33p0tbprYBN9DXVYwDl0UkTRrn2TK7SKPeBaxhzWlKBB5XrWTXIaofWZLICdTLumC+z1nsamqFp00" +
    "Ht4S+kcSlsXM+hTmq8uul4S0fUp5NZFhO5GtmtSZ0ycXRvXQHFooSGyRJlUD/uLjDjcbxr0Kv18fHlBLAwQUAAAACACUWD5d8z1D" +
    "RoYAAADhAAAAHwAAAHBwdC9fcmVscy9wcmVzZW50YXRpb24ueG1sLnJlbHNtjksKwzAMBa9idAArzqKLEifrbEsvYGI1CfUPyZQe" +
    "v6ZQaCAr8Z40g4bpHYN6EcuekwWjO5jG4UbB1VbIthdR7SKJha3WckWUZaPoROdCqW0emaOrLfKKxS1PtxL2XXdB/nfA0almb4Fn" +
    "b0DdHa9ULUjYPQl+h9FNDHjO9OdM/2Pw8Pz4AVBLAwQUAAAACACUWD5dSE1WEP8AAAA0AgAAFQAAAHBwdC9zbGlkZXMvc2xpZGUx" +
    "LnhtbJVSwU7DMAz9lShnNA8OHKq2E9w4gTR+IDTeGpE4UeK1K19P0lLBJCTG5cXJs5+e7dS7s7NiwJiMp0bebrZy19ahSlaLzFCq" +
    "QiN75lABpK5Hp9LGB6TMHXx0ivM1HiFETEisOKs4C3fb7T04ZUh+iahrRHRUo6HjRX3x0u2tnj2F14i4RAVp2IeXuETLGXrBU8BG" +
    "smGLEtoaVhJ+5vP50euprVUVCsQC3D5ENgfVscjdDAbHGsprwThjmFXWWlht/GnmLef/18sTWUN4I0bDvSAvCEehMc9MI3UG0y/e" +
    "LgWeCYVVH5Po+hO9i4BRaN+dXN7SVX3B97xhXQHM/6L9BFBLAwQUAAAACACUWD5d+mo9I+EAAAANAgAAFQAAAHBwdC9zbGlkZXMv" +
    "c2xpZGUyLnhtbJWR3WrDMAyFX8X4ulRZL3YRHBf2BGPdC3ixm6T4R9hq57797GRhKwzW3nyWkI50kMU+O8suJqYp+I4/bRu+lwLb" +
    "ZDUrFZ9a7PhIhC1A6kfjVNoGNL7UjiE6RSWNA2A0yXhSVKY4C7umeQanJs+/h6h7huioPic/3Oirl/5g9ewJ36MxS1TpLwd8jUu0" +
    "vDgyuqLpOE1kDQcpYC3C737KL0FfpVAtVsQKkm/B2nAmATWpjDNxFq8SWLf/6+Gj9D9qQYc+b1i2qRCR8h9ebgXohw07YQHq413O" +
    "4eeQsN4W5g+XX1BLAQIUAxQAAAAIAJRYPl1jzdLrngAAACABAAAUAAAAAAAAAAAAAACAAQAAAABwcHQvcHJlc2VudGF0aW9uLnht" +
    "bFBLAQIUAxQAAAAIAJRYPl3zPUNGhgAAAOEAAAAfAAAAAAAAAAAAAACAAdAAAABwcHQvX3JlbHMvcHJlc2VudGF0aW9uLnhtbC5y" +
    "ZWxzUEsBAhQDFAAAAAgAlFg+XUhNVhD/AAAANAIAABUAAAAAAAAAAAAAAIABkwEAAHBwdC9zbGlkZXMvc2xpZGUxLnhtbFBLAQIU" +
    "AxQAAAAIAJRYPl36aj0j4QAAAA0CAAAVAAAAAAAAAAAAAACAAcUCAABwcHQvc2xpZGVzL3NsaWRlMi54bWxQSwUGAAAAAAQABAAV" +
    "AQAA2QMAAAAA",
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
    { path: ARTIFACT_DEMO_PATHS.docx, id: "canvas-demo-docx", content: "" },
    { path: ARTIFACT_DEMO_PATHS.xlsx, id: "canvas-demo-xlsx", content: "" },
    { path: ARTIFACT_DEMO_PATHS.pptx, id: "canvas-demo-pptx", content: "" },
  ];

  return specs.flatMap((spec, index) =>
    artifactEvents(spec, startOffsetSeconds + index * 2, timestamp),
  );
}
