const PULL_REQUEST_NUMBER = /\/(?:pulls?|merge_requests|pr)\/(\d+)/i;

/** Short chip text for a linked PR URL (`#42`, or `PR` when the path has no number). */
export function pullRequestChipLabel(url: string): string {
  const match = url.trim().match(PULL_REQUEST_NUMBER);
  return match ? `#${match[1]}` : "PR";
}
