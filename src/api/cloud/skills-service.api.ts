import type { SkillInfo } from "#/types/settings";
import { getActiveBackend } from "../backend-registry/active-store";
import type { Backend } from "../backend-registry/types";
import { callCloudProxy } from "./proxy";

interface CloudSkillsPage {
  items: SkillInfo[];
  next_page_id: string | null;
}

const PAGE_LIMIT = 100;

function getActiveCloudBackend(): Backend {
  const active = getActiveBackend().backend;
  if (active.kind !== "cloud") {
    throw new Error("Cloud skills call requires a cloud backend.");
  }
  return active;
}

/**
 * Fetch the full list of skills from the cloud backend. The cloud endpoint is
 * paginated (page_id cursor); we walk all pages so the settings UI gets a
 * complete list in one call. The cloud SkillInfo shape
 * (name/type/source/triggers) matches the GUI's SkillInfo type, so items are
 * passed through unchanged.
 */
export async function fetchCloudSkills(): Promise<SkillInfo[]> {
  const backend = getActiveCloudBackend();

  const skills: SkillInfo[] = [];
  let pageId: string | null = null;

  do {
    const query = new URLSearchParams({ limit: String(PAGE_LIMIT) });
    if (pageId) query.set("page_id", pageId);

    const page = await callCloudProxy<CloudSkillsPage>({
      backend,
      method: "GET",
      path: `/api/v1/skills/search?${query.toString()}`,
    });

    skills.push(...(page.items ?? []));
    pageId = page.next_page_id;
  } while (pageId);

  return skills;
}
