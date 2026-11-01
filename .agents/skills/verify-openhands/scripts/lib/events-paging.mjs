// Paging for `conversation events`.
//
// The Agent Server's events/search answers at most 100 events per page and
// rejects a larger limit, so a conversation longer than that is read page by
// page (`next_page_id`) until `want` rows that pass `keep` (a kind or grep
// filter) are in hand or the pages run out. `fetchPage(pageId)` returns
// one page `{ items, next_page_id }`; every event read is returned, the
// caller filters and slices.

export const PAGE_LIMIT = 100;
export const MAX_PAGES = 200;

export async function collectEvents(fetchPage, want, keep = () => true) {
  const items = [];
  let pageId;
  let pages = 0;
  let kept = 0;
  do {
    const page = (await fetchPage(pageId)) ?? {};
    pages += 1;
    const got = Array.isArray(page.items) ? page.items : [];
    items.push(...got);
    kept += got.filter(keep).length;
    pageId = page.next_page_id ?? null;
  } while (pageId && kept < want && pages < MAX_PAGES);
  return { items, more: Boolean(pageId), pages };
}

// Image attachments in an LLM message's content blocks; `conversation events`
// rows show only text, so an image-only message would otherwise read blank.
export function countImages(content) {
  if (!Array.isArray(content)) return 0;
  let count = 0;
  for (const block of content) {
    if (!block || typeof block !== "object" || block.type !== "image") continue;
    count += Array.isArray(block.image_urls) ? block.image_urls.length : 1;
  }
  return count;
}
