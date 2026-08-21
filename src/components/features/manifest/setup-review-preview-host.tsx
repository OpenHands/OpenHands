import { useState } from "react";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ImportAutomationModal } from "#/components/features/automations/import-automation-modal";
import { SETUP_REGISTRY } from "#/manifests/manifest-sources";
import type { SetupFormValues } from "#/manifests/types";
import type { AutomationSpec } from "#/types/automation";
import { SetupDialog } from "./manifest-setup-dialog";
import { SETUP_REVIEW_PREVIEW_ENTRY_ID } from "./setup-review-preview";

const LONG_NAME =
  "Daily news digest for security advisories, quiet API deprecations, and feed-format changes that would actually stop a scheduled automation";

const LONG_PROMPT = [
  "Read the public RSS and Atom feeds on a schedule, skip items that are only link dumps or press-release reprints, and keep the ones that would change how we run automations.",
  "Prefer security advisories, breaking API or webhook changes, and feed-format shifts. Ignore product launches unless they retire something we depend on.",
  "Write a short digest: what changed, who it affects, and whether anyone needs to act before the next run.",
].join("\n\n");

const LONG_TOPIC =
  "security advisories that would actually stop a scheduled automation or leak a session key plus quiet deprecations in API versions webhook shapes cron semantics and feed formats that nobody notices until the next run fails";

const LONG_FEED =
  "https://feeds.example.com/research/daily-digest/v2/atom.xml?topics=ai-open-source-developer-tools&lookback=7d&include=full-text&format=atom&limit=500";

const LONG_PLUGIN =
  "a plugin whose display name is long enough that the chip has to wrap onto a second line instead of stretching the row";

const REVIEW_PREVIEW_VALUES: SetupFormValues = {
  feeds: [
    "https://news.ycombinator.com/rss",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.theverge.com/rss/index.xml",
    LONG_FEED,
  ].join("\n"),
  topics: [
    LONG_TOPIC,
    "artificial intelligence",
    "open source",
    "developer tools",
  ].join("\n"),
  prompt: LONG_PROMPT,
};

const IMPORT_PREVIEW_SPEC: AutomationSpec = {
  name: LONG_NAME,
  prompt: LONG_PROMPT,
  trigger: {
    type: "cron",
    schedule: "0 8 * * *",
    schedule_human: "Every day at 08:00",
    timezone: "UTC",
  },
  timezone: "UTC",
  enabled: true,
  plugins: ["rss-reader", LONG_PLUGIN],
};

/**
 * Design host: Review (ready-to-use setup) and Import preview side by side.
 */
export function SetupReviewPreviewHost() {
  const [dismissed, setDismissed] = useState(false);
  const entry = SETUP_REGISTRY.findById(SETUP_REVIEW_PREVIEW_ENTRY_ID);
  if (!entry || dismissed) return null;

  return (
    <ModalBackdrop onClose={() => setDismissed(true)}>
      <div className="flex max-h-[90vh] max-w-[96vw] items-start gap-6 overflow-x-auto">
        <SetupDialog
          key="review-preview-long-content"
          entry={{ ...entry, name: LONG_NAME }}
          initialStep="review"
          preview
          previewValues={REVIEW_PREVIEW_VALUES}
          embedded
          onClose={() => setDismissed(true)}
        />
        <ImportAutomationModal
          key="import-preview-long-content"
          isOpen
          embedded
          spec={IMPORT_PREVIEW_SPEC}
          isImporting={false}
          onClose={() => setDismissed(true)}
          onImport={() => setDismissed(true)}
          onFile={() => undefined}
        />
      </div>
    </ModalBackdrop>
  );
}
