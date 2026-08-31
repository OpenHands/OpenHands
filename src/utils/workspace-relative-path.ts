/**
 * Workspace-relative path helpers for Files drawer / workspace file APIs.
 *
 * Implementation lives in path-utils (`toFilesTabPath`) so chat links,
 * canvas_ui navigation, and markdown artifact View share one normalizer.
 */
export { toFilesTabPath as toWorkspaceRelativePath } from "./path-utils";
