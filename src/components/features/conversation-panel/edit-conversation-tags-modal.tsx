import React from "react";
import { useTranslation } from "react-i18next";
import { X } from "lucide-react";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BrandButton } from "../settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import {
  getDisplayConversationTags,
  RESERVED_CONVERSATION_TAG_KEYS,
} from "#/api/agent-server-adapter";

/** Backend rule for tag keys (see `ConversationInfo.tags` in the SDK). */
export const CONVERSATION_TAG_KEY_PATTERN = /^[a-z0-9]+$/;
/** Backend cap on tag values (see `ConversationInfo.tags` in the SDK). */
export const CONVERSATION_TAG_VALUE_MAX_LENGTH = 256;

interface TagRow {
  key: string;
  value: string;
}

/**
 * Merges the user's edited display tags with every entry the display helper
 * drops from the current map (reserved/internal keys, empty or non-string
 * values), so the replace-all PATCH never silently discards internal tags.
 */
export function mergeConversationTagEdits(
  currentTags: Record<string, string> | null | undefined,
  editedUserTags: readonly (readonly [string, string])[],
): Record<string, string> {
  const merged: Record<string, string> = {};
  for (const [key, value] of Object.entries(currentTags ?? {})) {
    if (
      RESERVED_CONVERSATION_TAG_KEYS.has(key.trim().toLowerCase()) ||
      typeof value !== "string" ||
      value.trim().length === 0
    ) {
      merged[key] = value;
    }
  }
  for (const [key, value] of editedUserTags) {
    merged[key] = value;
  }
  return merged;
}

interface EditConversationTagsModalProps {
  /** The conversation's complete server-side tag map (including internals). */
  tags: Record<string, string> | null | undefined;
  /** Called with the merged complete map (user edits + preserved internals). */
  onConfirm: (mergedTags: Record<string, string>) => void;
  onCancel: () => void;
}

export function EditConversationTagsModal({
  tags,
  onConfirm,
  onCancel,
}: EditConversationTagsModalProps) {
  const { t } = useTranslation("openhands");
  const [rows, setRows] = React.useState<TagRow[]>(() =>
    getDisplayConversationTags(tags).map(([key, value]) => ({ key, value })),
  );
  const [newKey, setNewKey] = React.useState("");
  const [newValue, setNewValue] = React.useState("");
  const [errorKey, setErrorKey] = React.useState<I18nKey | null>(null);

  const handleAdd = () => {
    const key = newKey.trim();
    const value = newValue.trim();
    if (!CONVERSATION_TAG_KEY_PATTERN.test(key)) {
      setErrorKey(I18nKey.CONVERSATION$TAG_KEY_INVALID);
      return;
    }
    if (RESERVED_CONVERSATION_TAG_KEYS.has(key)) {
      setErrorKey(I18nKey.CONVERSATION$TAG_KEY_RESERVED);
      return;
    }
    if (rows.some((row) => row.key === key)) {
      setErrorKey(I18nKey.CONVERSATION$TAG_KEY_DUPLICATE);
      return;
    }
    if (
      value.length === 0 ||
      value.length > CONVERSATION_TAG_VALUE_MAX_LENGTH
    ) {
      setErrorKey(I18nKey.CONVERSATION$TAG_VALUE_INVALID);
      return;
    }
    setRows((prev) => [...prev, { key, value }]);
    setNewKey("");
    setNewValue("");
    setErrorKey(null);
  };

  const handleRemove = (key: string) => {
    setRows((prev) => prev.filter((row) => row.key !== key));
  };

  const handleConfirm = () => {
    onConfirm(
      mergeConversationTagEdits(
        tags,
        rows.map((row) => [row.key, row.value] as const),
      ),
    );
  };

  return (
    <ModalBackdrop onClose={onCancel}>
      <ModalBody
        className="items-start border border-[var(--oh-border)]"
        testID="edit-conversation-tags-modal"
      >
        <div className="flex flex-col gap-2">
          <BaseModalTitle title={t(I18nKey.CONVERSATION$EDIT_TAGS)} />
          <BaseModalDescription>
            {t(I18nKey.CONVERSATION$EDIT_TAGS_DESCRIPTION)}
          </BaseModalDescription>
        </div>

        <div
          className="flex w-full flex-col gap-2"
          onClick={(event) => event.stopPropagation()}
        >
          {rows.length > 0 ? (
            <ul
              className="flex w-full flex-col gap-1"
              data-testid="edit-tags-rows"
            >
              {rows.map((row) => (
                <li
                  key={row.key}
                  className="flex items-center gap-2 rounded-md border border-[var(--oh-border)] px-2 py-1"
                  data-testid={`edit-tag-row-${row.key}`}
                >
                  <span className="min-w-0 flex-1 truncate text-xs text-white">
                    {row.key}={row.value}
                  </span>
                  <button
                    type="button"
                    aria-label={t(I18nKey.CONVERSATION$REMOVE_TAG)}
                    data-testid={`remove-tag-${row.key}`}
                    onClick={() => handleRemove(row.key)}
                    className="shrink-0 text-[var(--oh-muted)] hover:text-white"
                  >
                    <X className="h-3.5 w-3.5" aria-hidden />
                  </button>
                </li>
              ))}
            </ul>
          ) : null}

          <div className="flex w-full items-center gap-2">
            <input
              type="text"
              value={newKey}
              placeholder={t(I18nKey.CONVERSATION$TAG_KEY_PLACEHOLDER)}
              aria-label={t(I18nKey.CONVERSATION$TAG_KEY_PLACEHOLDER)}
              data-testid="new-tag-key-input"
              onChange={(event) => setNewKey(event.target.value)}
              className="w-28 shrink-0 rounded-md border border-[var(--oh-border)] bg-base-secondary px-2 py-1 text-xs text-white placeholder:text-[var(--oh-muted)]"
            />
            <input
              type="text"
              value={newValue}
              placeholder={t(I18nKey.CONVERSATION$TAG_VALUE_PLACEHOLDER)}
              aria-label={t(I18nKey.CONVERSATION$TAG_VALUE_PLACEHOLDER)}
              data-testid="new-tag-value-input"
              onChange={(event) => setNewValue(event.target.value)}
              className="min-w-0 flex-1 rounded-md border border-[var(--oh-border)] bg-base-secondary px-2 py-1 text-xs text-white placeholder:text-[var(--oh-muted)]"
            />
            <BrandButton
              type="button"
              variant="secondary"
              onClick={handleAdd}
              testId="add-tag-button"
            >
              {t(I18nKey.BUTTON$ADD)}
            </BrandButton>
          </div>

          {errorKey ? (
            <p
              role="alert"
              data-testid="edit-tags-error"
              className="text-xs text-red-400"
            >
              {t(errorKey)}
            </p>
          ) : null}
        </div>

        <div
          className="flex justify-end gap-2 w-full"
          onClick={(event) => event.stopPropagation()}
        >
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onCancel}
            testId="cancel-button"
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            onClick={handleConfirm}
            testId="confirm-button"
          >
            {t(I18nKey.BUTTON$SAVE)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
