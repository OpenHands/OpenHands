// Paging for `conversation events`.
//
// The Agent Server's events/search answers at most 100 events per page and
// rejects a larger limit, so a conversation longer than that is read page by
// page (`next_page_id`) until the rows asked for are in hand or the pages run
// out. `fetchPage(pageId)` returns one page `{ items, next_page_id }`.

export const PAGE_LIMIT = 100;
const MAX_PAGES = 200;

export async function collectEvents(fetchPage, want) {
  const items = [];
  let pageId;
  let pages = 0;
  do {
    const page = (await fetchPage(pageId)) ?? {};
    pages += 1;
    items.push(...(Array.isArray(page.items) ? page.items : []));
    pageId = page.next_page_id ?? null;
  } while (pageId && items.length < want && pages < MAX_PAGES);
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
