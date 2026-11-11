import { http, HttpResponse } from "msw";
import type { MetaProfile } from "@openhands/typescript-client";

const profiles = new Map<string, MetaProfile>();
let activeProfile: string | null = null;

export function resetMockMetaProfiles(): void {
  profiles.clear();
  activeProfile = null;
}

function missing(name: string) {
  return HttpResponse.json(
    { detail: `Meta-profile '${name}' not found` },
    { status: 404 },
  );
}

export const META_PROFILES_HANDLERS = [
  http.get("*/api/meta-profiles", () =>
    HttpResponse.json({
      meta_profiles: [...profiles.entries()].map(([name, config]) => ({
        name,
        classifier_model: config.classifier_model,
        num_classes: config.classes?.length ?? 0,
      })),
      active_meta_profile: activeProfile,
    }),
  ),
  http.get("*/api/meta-profiles/:name", ({ params }) => {
    const name = String(params.name);
    const config = profiles.get(name);
    return config ? HttpResponse.json({ name, config }) : missing(name);
  }),
  http.post("*/api/meta-profiles/:name", async ({ params, request }) => {
    const name = String(params.name);
    const config = (await request.json()) as Partial<MetaProfile>;
    if (
      !config?.classifier_model?.trim() ||
      (!config.classes?.length &&
        !config.prompt_template?.includes("{{ instance_text }}"))
    ) {
      return HttpResponse.json(
        {
          detail:
            "A classifier and valid routing prompt or classes are required",
        },
        { status: 422 },
      );
    }
    profiles.set(name, config as MetaProfile);
    return HttpResponse.json({ name, message: "Meta-profile saved" });
  }),
  http.post("*/api/meta-profiles/:name/activate", ({ params }) => {
    const name = String(params.name);
    if (!profiles.has(name)) return missing(name);
    activeProfile = name;
    return HttpResponse.json({ name, message: "Meta-profile activated" });
  }),
  http.delete("*/api/meta-profiles/:name", ({ params }) => {
    const name = String(params.name);
    if (!profiles.delete(name)) return missing(name);
    if (activeProfile === name) activeProfile = null;
    return HttpResponse.json({ name, message: "Meta-profile deleted" });
  }),
];
