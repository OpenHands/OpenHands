/**
 * Removes items whose `id` was already seen, keeping the first copy and the
 * original order. Paged lists use it: a row can come back on a later page
 * when the list shifts between requests.
 */
export function uniqueById<T extends { id: string }>(items: T[]): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  });
}
