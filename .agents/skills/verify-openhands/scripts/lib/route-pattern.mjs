// A route path from src/routes.ts as a pattern that finds it in the map's
// prose, where paths are written in backticks with a placeholder or a concrete
// value for each parameter: `/conversations/<id>`, `/conversations/abc-123`
// or `/conversations/:conversationId`. Optional parameters may be left out and
// a trailing splat matches whatever follows.
const PARAM = "(?::\\w+|<[^>`]+>|[\\w.-]+)";

export function routePattern(path) {
  const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  let source = "";
  for (const segment of path
    .replace(/\/?\*$/, "")
    .split("/")
    .filter(Boolean)) {
    const param = /^:\w+(\?)?$/.exec(segment);
    if (!param) source += `/${escape(segment)}`;
    else source += param[1] ? `(?:/${PARAM})?` : `/${PARAM}`;
  }
  return new RegExp(`\`${source || "/"}(?=[\`/?#\\s)]|$)`);
}
