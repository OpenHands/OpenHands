import { type KeyboardEvent, useState } from "react";
import { X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import { useUpdateAutomationTags } from "#/hooks/query/use-automations";
import { AutomationTagPill } from "#/components/features/automations/automation-tag-pill";
import {
  AUTOMATION_TAG_MAX_COUNT,
  AUTOMATION_TAG_MAX_LENGTH,
  type AutomationTagError,
  getAutomationTagError,
  normalizeAutomationTag,
} from "#/utils/automation-tags";

const TAG_ERROR_KEYS: Record<AutomationTagError, I18nKey> = {
  invalid: I18nKey.AUTOMATIONS$TAG_ERROR_INVALID,
  "too-long": I18nKey.AUTOMATIONS$TAG_ERROR_TOO_LONG,
  "too-many": I18nKey.AUTOMATIONS$TAG_ERROR_TOO_MANY,
};

interface AutomationTagsEditorProps {
  automation: Automation;
  /** Only the creator may change tags; everyone else sees them read-only. */
  canEdit: boolean;
}

/**
 * The automation's tags under the detail header. The creator edits them in
 * place and saves the whole list in one PATCH, so a tag the service rejects
 * never leaves the others half-saved.
 */
export function AutomationTagsEditor({
  automation,
  canEdit,
}: AutomationTagsEditorProps) {
  const { t } = useTranslation("openhands");
  const updateTags = useUpdateAutomationTags();
  const [draft, setDraft] = useState<string[] | null>(null);
  const [input, setInput] = useState("");
  const [error, setError] = useState<AutomationTagError | null>(null);

  // An automation service without tag support sends no `tags` field, and
  // would reject one in a PATCH.
  if (!automation.tags) return null;
  const savedTags = automation.tags;
  if (savedTags.length === 0 && !canEdit) return null;

  const closeEditor = () => {
    setDraft(null);
    setInput("");
    setError(null);
  };

  const addInputTag = (tags: string[]): string[] | null => {
    const tag = normalizeAutomationTag(input);
    if (!tag) return tags;
    const tagError = getAutomationTagError(tags, tag);
    setError(tagError);
    if (tagError) return null;
    setInput("");
    return tags.includes(tag) ? tags : [...tags, tag];
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== "Enter" || !draft) return;
    event.preventDefault();
    const next = addInputTag(draft);
    if (next) setDraft(next);
  };

  const handleSave = () => {
    if (!draft) return;
    // A tag still in the input counts, so Save does not drop it silently.
    const tags = addInputTag(draft);
    if (!tags) return;
    updateTags.mutate({ id: automation.id, tags }, { onSuccess: closeEditor });
  };

  if (!draft) {
    return (
      <div
        data-testid="automation-tags"
        className="flex flex-wrap items-center gap-1.5"
      >
        {savedTags.map((tag) => (
          <AutomationTagPill key={tag} tag={tag} />
        ))}
        {canEdit ? (
          <button
            type="button"
            data-testid="automation-tags-edit"
            onClick={() => setDraft(savedTags)}
            className="text-xs text-muted hover:text-foreground"
          >
            {t(I18nKey.AUTOMATIONS$EDIT_TAGS)}
          </button>
        ) : null}
      </div>
    );
  }

  return (
    <div data-testid="automation-tags-editor" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {draft.map((tag) => (
          <AutomationTagPill key={tag} tag={tag}>
            <button
              type="button"
              aria-label={t(I18nKey.AUTOMATIONS$REMOVE_TAG, { tag })}
              onClick={() => setDraft(draft.filter((other) => other !== tag))}
              className="text-muted hover:text-foreground"
            >
              <X className="size-3" aria-hidden="true" />
            </button>
          </AutomationTagPill>
        ))}
        <input
          data-testid="automation-tags-input"
          value={input}
          onChange={(event) => {
            setInput(event.target.value);
            setError(null);
          }}
          onKeyDown={handleKeyDown}
          placeholder={t(I18nKey.AUTOMATIONS$ADD_TAG_PLACEHOLDER)}
          aria-label={t(I18nKey.AUTOMATIONS$ADD_TAG_PLACEHOLDER)}
          aria-invalid={error !== null}
          maxLength={AUTOMATION_TAG_MAX_LENGTH * 2}
          className="min-w-40 flex-1 rounded-md border border-border bg-transparent px-2 py-1 text-xs text-content focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        />
      </div>
      {error ? (
        <p role="alert" className="text-xs text-feedback-error">
          {t(TAG_ERROR_KEYS[error], {
            max:
              error === "too-many"
                ? AUTOMATION_TAG_MAX_COUNT
                : AUTOMATION_TAG_MAX_LENGTH,
          })}
        </p>
      ) : null}
      <div className="flex items-center gap-2">
        <button
          type="button"
          data-testid="automation-tags-save"
          onClick={handleSave}
          disabled={updateTags.isPending}
          className="rounded-md border border-border px-3 py-1 text-xs font-medium text-content hover:bg-surface-raised disabled:cursor-not-allowed disabled:opacity-60"
        >
          {t(I18nKey.AUTOMATIONS$SAVE)}
        </button>
        <button
          type="button"
          onClick={closeEditor}
          disabled={updateTags.isPending}
          className="text-xs text-muted hover:text-foreground"
        >
          {t(I18nKey.AUTOMATIONS$CANCEL)}
        </button>
      </div>
    </div>
  );
}
