// ---------------------------------------------------------------------------
// All verified-model and verified-provider lists are now served by the
// backend via ``GET /api/options/models`` (see ``ModelsResponse``).
//
// This file only keeps the compile-time fallback default used when no API
// response is available yet (e.g. the initial settings page render).
// ---------------------------------------------------------------------------

/** Fallback default model shown before the API responds. */
export const DEFAULT_OPENHANDS_MODEL = "openhands/claude-opus-4-5-20251101";
