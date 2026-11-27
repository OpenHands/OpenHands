import { RegisteredVisualizer } from "./define";
import { bashVisualizer } from "./bash/bash";
import { fileEditorVisualizer } from "./file-editor/file-editor";
import { searchVisualizer } from "./search/search";
import { taskVisualizer } from "./task/task";

/**
 * Tool visualizers render a tool call's action / observation card body as React
 * components instead of markdown. Unregistered tools keep using the markdown
 * pipeline (see `dispatcher.ts`), so this list can grow one tool at a time.
 *
 * To add a visualizer:
 *   1. Create `tool-visualizers/<name>/<name>.tsx` exporting
 *      `defineVisualizer({ actionKinds, observationKinds, Body })`.
 *   2. Add one import below and one entry to `ALL`.
 *   3. Add `tool-visualizers/<name>/<name>.test.tsx` (render with fixtures).
 *
 * TypeScript does the rest: `actionKinds` autocompletes from the `Action`
 * union, `Body` receives the narrowed event types, and unknown kinds are
 * compile errors.
 */
const ALL: RegisteredVisualizer[] = [
  bashVisualizer,
  fileEditorVisualizer,
  searchVisualizer,
  taskVisualizer,
];

const indexByKind = (
  kindsOf: (visualizer: RegisteredVisualizer) => string[] | undefined,
): Map<string, RegisteredVisualizer> =>
  new Map(
    ALL.flatMap((visualizer) =>
      (kindsOf(visualizer) ?? []).map((kind) => [kind, visualizer] as const),
    ),
  );

export const actionVisualizers = indexByKind((v) => v.actionKinds);
export const observationVisualizers = indexByKind((v) => v.observationKinds);
