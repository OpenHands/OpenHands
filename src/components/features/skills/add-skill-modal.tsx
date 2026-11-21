import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  ADD_SKILL_DOCS_URL,
  ADD_SKILL_EXAMPLE_COMMAND,
} from "#/constants/skills-docs";
import { I18nKey } from "#/i18n/declaration";
import type { SkillInfo } from "#/types/settings";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import { formControlSettingsFieldClassName } from "#/utils/form-control-classes";
import CheckmarkIcon from "#/icons/checkmark.svg?react";
import CopyIcon from "#/icons/copy.svg?react";
import { copyToClipboard } from "#/utils/copy-to-clipboard";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import SkillsService from "#/api/skills-service";
import { parseSkillFrontmatter } from "#/utils/custom-skills-storage";

interface AddSkillModalProps {
  onClose: () => void;
  initialSkill?: SkillInfo | null;
  onSaved?: (skill: SkillInfo) => void;
}

const ADD_SKILL_STEP_KEYS = [
  I18nKey.SETTINGS$SKILLS_ADD_MODAL_STEP_1,
  I18nKey.SETTINGS$SKILLS_ADD_MODAL_STEP_2,
  I18nKey.SETTINGS$SKILLS_ADD_MODAL_STEP_3,
  I18nKey.SETTINGS$SKILLS_ADD_MODAL_STEP_4,
  I18nKey.SETTINGS$SKILLS_ADD_MODAL_STEP_5,
] as const;

const ADD_SKILL_INLINE_CODE_COMPONENTS = {
  cmd: <InlineCodeChip />,
  path: <InlineCodeChip />,
  env: <InlineCodeChip />,
};

function InlineCodeChip({ children }: { children?: React.ReactNode }) {
  return (
    <code
      className={cn(
        "mx-0.5 inline-block rounded-sm border border-border-subtle",
        "bg-surface-raised px-1.5 py-0.5 align-baseline font-mono text-2xs text-contrast",
      )}
    >
      {children}
    </code>
  );
}

function AddSkillExampleBlock() {
  const { t } = useTranslation("openhands");
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    if (await copyToClipboard(ADD_SKILL_EXAMPLE_COMMAND)) {
      setCopied(true);
    } else {
      displayErrorToast(t(I18nKey.CHAT_INTERFACE$CHAT_MESSAGE_COPY_FAILED));
    }
  };

  React.useEffect(() => {
    if (!copied) return undefined;
    const timeout = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(timeout);
  }, [copied]);

  return (
    <div className="relative">
      <pre
        data-testid="add-skill-modal-example"
        className={cn(
          "overflow-x-auto rounded-sm border border-border-subtle",
          "bg-surface-raised p-2 pr-10 text-xs text-contrast",
        )}
      >
        {ADD_SKILL_EXAMPLE_COMMAND}
      </pre>
      <button
        type="button"
        data-testid="add-skill-modal-example-copy"
        aria-label={t(copied ? I18nKey.BUTTON$COPIED : I18nKey.BUTTON$COPY)}
        disabled={copied}
        onClick={handleCopy}
        className={cn(
          "absolute right-2 top-2 cursor-pointer rounded-sm border border-border-subtle",
          "bg-base-secondary p-1 text-tertiary-alt transition-colors",
          "hover:bg-surface hover:text-contrast disabled:cursor-default [&_path]:fill-current",
        )}
      >
        {copied ? (
          <CheckmarkIcon width={14} height={14} />
        ) : (
          <CopyIcon width={14} height={14} />
        )}
      </button>
    </div>
  );
}

function AddSkillTransParagraph({
  i18nKey,
}: {
  i18nKey:
    | typeof I18nKey.SETTINGS$SKILLS_ADD_MODAL_CHAT_BODY
    | typeof I18nKey.SETTINGS$SKILLS_ADD_MODAL_STORAGE_BODY
    | typeof I18nKey.SETTINGS$SKILLS_ADD_MODAL_PRIVATE_REPOS;
}) {
  return (
    <p className="text-xs leading-relaxed text-tertiary-light">
      <Trans
        ns="openhands"
        i18nKey={i18nKey}
        components={ADD_SKILL_INLINE_CODE_COMPONENTS}
      />
    </p>
  );
}

export function AddSkillModal({
  onClose,
  initialSkill,
  onSaved,
}: AddSkillModalProps) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();

  const isEditing = Boolean(initialSkill);
  const [activeTab, setActiveTab] = React.useState<"custom" | "repo">("custom");

  const [name, setName] = React.useState(initialSkill?.name ?? "");
  const [triggers, setTriggers] = React.useState(
    initialSkill?.triggers?.join(", ") ?? "",
  );
  const [description, setDescription] = React.useState(
    initialSkill?.description ?? "",
  );
  const [content, setContent] = React.useState(initialSkill?.content ?? "");

  const [nameError, setNameError] = React.useState<string | null>(null);
  const [contentError, setContentError] = React.useState<string | null>(null);

  const handleContentChange = (
    event: React.ChangeEvent<HTMLTextAreaElement>,
  ) => {
    const newContent = event.target.value;
    setContent(newContent);
    if (contentError) setContentError(null);

    const parsed = parseSkillFrontmatter(newContent);
    if (parsed.name && !name) {
      setName(parsed.name);
      if (nameError) setNameError(null);
    }
    if (parsed.description && !description) {
      setDescription(parsed.description);
    }
    if (parsed.triggers?.length && !triggers) {
      setTriggers(parsed.triggers.join(", "));
    }
  };

  const handleSave = () => {
    const trimmedName = name.trim().toLowerCase();
    if (!trimmedName) {
      setNameError(t(I18nKey.SETTINGS$SKILLS_CUSTOM_NAME_REQUIRED));
      return;
    }
    if (!/^[a-z0-9_-]+$/.test(trimmedName)) {
      setNameError(t(I18nKey.SETTINGS$SKILLS_CUSTOM_NAME_INVALID));
      return;
    }
    if (!content.trim()) {
      setContentError(t(I18nKey.SETTINGS$SKILLS_CUSTOM_CONTENT_REQUIRED));
      return;
    }

    const parsedTriggers = triggers
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);

    try {
      const saved = SkillsService.saveCustomSkill({
        name: trimmedName,
        description: description.trim() || undefined,
        triggers:
          parsedTriggers.length > 0 ? parsedTriggers : [`/${trimmedName}`],
        content: content.trim(),
      });

      queryClient.invalidateQueries({ queryKey: ["skills"] });
      queryClient.invalidateQueries({ queryKey: ["skill-catalog"] });
      queryClient.invalidateQueries({ queryKey: ["conversation"] });

      displaySuccessToast(t(I18nKey.SETTINGS$SKILLS_CUSTOM_SAVE_SUCCESS));
      onSaved?.(saved);
      onClose();
    } catch (error) {
      displayErrorToast(error instanceof Error ? error.message : String(error));
    }
  };

  const tabButtonClass = (isActive: boolean) =>
    cn(
      "cursor-pointer border-b-2 px-3 py-2 text-sm font-medium transition-colors",
      isActive
        ? "border-contrast text-contrast"
        : "border-transparent text-muted hover:text-contrast",
    );

  const modalTitle = isEditing
    ? t(I18nKey.SETTINGS$SKILLS_CUSTOM_EDIT_TITLE)
    : t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_TITLE);

  return (
    <ModalBackdrop onClose={onClose} aria-label={modalTitle}>
      <div
        data-testid="add-skill-modal"
        className="relative flex w-150 extension-dialog-viewport flex-col rounded-xl border border-border bg-base-secondary"
      >
        <ModalCloseButton onClose={onClose} testId="add-skill-modal-close" />
        <header className="flex-shrink-0 px-6 pb-2 pt-6">
          <h2 className={cn("pr-6", modalTitleLgClassName)}>{modalTitle}</h2>
          {!isEditing ? (
            <div
              role="tablist"
              className="mt-4 flex border-b border-border-subtle"
            >
              <button
                type="button"
                role="tab"
                data-testid="add-skill-modal-tab-custom"
                aria-selected={activeTab === "custom"}
                className={tabButtonClass(activeTab === "custom")}
                onClick={() => setActiveTab("custom")}
              >
                {t(I18nKey.SETTINGS$SKILLS_TAB_CREATE_CUSTOM)}
              </button>
              <button
                type="button"
                role="tab"
                data-testid="add-skill-modal-tab-repo"
                aria-selected={activeTab === "repo"}
                className={tabButtonClass(activeTab === "repo")}
                onClick={() => setActiveTab("repo")}
              >
                {t(I18nKey.SETTINGS$SKILLS_TAB_IMPORT_REPO)}
              </button>
            </div>
          ) : null}
        </header>

        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-6 py-2 custom-scrollbar">
          {activeTab === "custom" ? (
            <div className="flex flex-col gap-4">
              <label className="flex flex-col gap-1.5 text-xs text-foreground">
                <span className="font-medium">
                  {t(I18nKey.SETTINGS$SKILLS_CUSTOM_NAME_LABEL)} *
                </span>
                <input
                  type="text"
                  data-testid="custom-skill-name-input"
                  value={name}
                  disabled={isEditing}
                  placeholder={t(
                    I18nKey.SETTINGS$SKILLS_CUSTOM_NAME_PLACEHOLDER,
                  )}
                  onChange={(e) => {
                    setName(e.target.value);
                    if (nameError) setNameError(null);
                  }}
                  className={cn(
                    formControlSettingsFieldClassName,
                    "w-full rounded-md border border-border-subtle bg-surface-raised px-3 py-2 text-sm text-contrast",
                    nameError && "border-danger",
                  )}
                />
                {nameError ? (
                  <span
                    data-testid="custom-skill-name-error"
                    className="text-2xs text-danger"
                  >
                    {nameError}
                  </span>
                ) : null}
              </label>

              <label className="flex flex-col gap-1.5 text-xs text-foreground">
                <span className="font-medium">
                  {t(I18nKey.SETTINGS$SKILLS_CUSTOM_TRIGGERS_LABEL)}
                </span>
                <input
                  type="text"
                  data-testid="custom-skill-triggers-input"
                  value={triggers}
                  placeholder={t(
                    I18nKey.SETTINGS$SKILLS_CUSTOM_TRIGGERS_PLACEHOLDER,
                  )}
                  onChange={(e) => setTriggers(e.target.value)}
                  className={cn(
                    formControlSettingsFieldClassName,
                    "w-full rounded-md border border-border-subtle bg-surface-raised px-3 py-2 text-sm text-contrast",
                  )}
                />
              </label>

              <label className="flex flex-col gap-1.5 text-xs text-foreground">
                <span className="font-medium">
                  {t(I18nKey.SETTINGS$SKILLS_CUSTOM_DESCRIPTION_LABEL)}
                </span>
                <textarea
                  data-testid="custom-skill-description-input"
                  value={description}
                  rows={4}
                  placeholder={t(
                    I18nKey.SETTINGS$SKILLS_CUSTOM_DESCRIPTION_PLACEHOLDER,
                  )}
                  onChange={(e) => setDescription(e.target.value)}
                  className={cn(
                    formControlSettingsFieldClassName,
                    "w-full rounded-md border border-border-subtle bg-surface-raised px-3 py-2 text-sm text-contrast resize-y custom-scrollbar",
                  )}
                />
              </label>

              <label className="flex flex-col gap-1.5 text-xs text-foreground">
                <span className="font-medium">
                  {t(I18nKey.SETTINGS$SKILLS_CUSTOM_CONTENT_LABEL)} *
                </span>
                <textarea
                  data-testid="custom-skill-content-input"
                  value={content}
                  rows={10}
                  placeholder={t(
                    I18nKey.SETTINGS$SKILLS_CUSTOM_CONTENT_PLACEHOLDER,
                  )}
                  onChange={handleContentChange}
                  className={cn(
                    formControlSettingsFieldClassName,
                    "w-full rounded-md border border-border-subtle bg-surface-raised px-3 py-2 font-mono text-xs text-contrast resize-y custom-scrollbar",
                    contentError && "border-danger",
                  )}
                />
                {contentError ? (
                  <span
                    data-testid="custom-skill-content-error"
                    className="text-2xs text-danger"
                  >
                    {contentError}
                  </span>
                ) : null}
              </label>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              <p className="text-sm text-tertiary-light">
                {t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_INTRO)}
              </p>
              <section className="flex flex-col gap-2">
                <h3 className="text-sm font-semibold text-foreground">
                  {t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_CHAT_TITLE)}
                </h3>
                <AddSkillTransParagraph
                  i18nKey={I18nKey.SETTINGS$SKILLS_ADD_MODAL_CHAT_BODY}
                />
                <p className="text-xs text-tertiary-light">
                  {t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_EXAMPLE_LABEL)}
                </p>
                <AddSkillExampleBlock />
                <ol className="list-decimal space-y-1 pl-4 text-xs leading-relaxed text-tertiary-light">
                  {ADD_SKILL_STEP_KEYS.map((key) => (
                    <li key={key}>
                      {key === I18nKey.SETTINGS$SKILLS_ADD_MODAL_STEP_3 ? (
                        <Trans
                          ns="openhands"
                          i18nKey={key}
                          components={ADD_SKILL_INLINE_CODE_COMPONENTS}
                        />
                      ) : (
                        t(key)
                      )}
                    </li>
                  ))}
                </ol>
              </section>

              <section className="flex flex-col gap-1">
                <h3 className="text-sm font-semibold text-foreground">
                  {t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_URL_FORMATS_TITLE)}
                </h3>
                <p className="text-xs leading-relaxed text-tertiary-light">
                  {t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_URL_FORMATS)}
                </p>
              </section>

              <section className="flex flex-col gap-1">
                <h3 className="text-sm font-semibold text-foreground">
                  {t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_STORAGE_TITLE)}
                </h3>
                <AddSkillTransParagraph
                  i18nKey={I18nKey.SETTINGS$SKILLS_ADD_MODAL_STORAGE_BODY}
                />
              </section>

              <AddSkillTransParagraph
                i18nKey={I18nKey.SETTINGS$SKILLS_ADD_MODAL_PRIVATE_REPOS}
              />

              <a
                href={ADD_SKILL_DOCS_URL}
                target="_blank"
                rel="noreferrer"
                data-testid="add-skill-modal-docs-link"
                className="self-start text-xs text-muted transition-colors hover:text-contrast hover:underline"
              >
                {t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_VIEW_DOCS)}
              </a>
            </div>
          )}
        </div>

        <footer className="flex flex-shrink-0 justify-end gap-2 px-6 pb-6 pt-4">
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onClose}
            testId="add-skill-modal-dismiss"
          >
            {t(
              activeTab === "custom"
                ? I18nKey.BUTTON$CANCEL
                : I18nKey.BUTTON$CLOSE,
            )}
          </BrandButton>
          {activeTab === "custom" ? (
            <BrandButton
              type="button"
              variant="primary"
              onClick={handleSave}
              testId="save-custom-skill-button"
            >
              {t(I18nKey.BUTTON$SAVE)}
            </BrandButton>
          ) : null}
        </footer>
      </div>
    </ModalBackdrop>
  );
}
