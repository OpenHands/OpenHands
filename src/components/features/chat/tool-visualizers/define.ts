import React from "react";
import {
  Action,
  Observation,
  ActionEvent,
  ObservationEvent,
} from "#/types/agent-server/core";

type ActionKind = Action["kind"];
type ObservationKind = Observation["kind"];
type ActionByKind<K extends ActionKind> = Extract<Action, { kind: K }>;
type ObservationByKind<K extends ObservationKind> = Extract<
  Observation,
  { kind: K }
>;

/**
 * Props a visualizer `Body` receives. The chat uses a two-card layout, so each
 * card passes only the half it owns: the action card sets `action`, and the
 * observation card sets `observation` (plus the originating `action` when it
 * can be resolved). A `Body` must render whichever halves are present.
 */
export interface VisualizerProps<A extends Action, O extends Observation> {
  action?: ActionEvent<A>;
  observation?: ObservationEvent<O>;
}

/**
 * Author-facing visualizer shape. `actionKinds` / `observationKinds` are
 * narrowed string-literal arrays, so `Body` is type-checked against the exact
 * `Action` / `Observation` members for those kinds — no casts in the body.
 */
export interface ToolVisualizer<
  AK extends ActionKind,
  OK extends ObservationKind,
> {
  actionKinds: AK[];
  observationKinds?: OK[];
  Body: React.FC<VisualizerProps<ActionByKind<AK>, ObservationByKind<OK>>>;
}

/**
 * Type-erased shape stored in the registry so visualizers with different kinds
 * can share one `Map`. Lookups re-narrow by the registered `kind`.
 */
export interface RegisteredVisualizer {
  actionKinds: string[];
  observationKinds?: string[];
  Body: React.FC<VisualizerProps<Action, Observation>>;
}

/**
 * Identity helper that infers the narrow generics from `actionKinds` /
 * `observationKinds` (giving `Body` precise prop types and autocompleted
 * kinds) and returns the erased form for the registry.
 */
export const defineVisualizer = <
  AK extends ActionKind,
  OK extends ObservationKind = never,
>(
  visualizer: ToolVisualizer<AK, OK>,
): RegisteredVisualizer => visualizer as unknown as RegisteredVisualizer;
