import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  ADD_SKILL_DOCS_URL,
  ADD_SKILL_EXAMPLE_COMMAND,
} from "#/constants/skills-docs";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import CheckmarkIcon from "#/icons/checkmark.svg?react";
import CopyIcon from "#/icons/copy.svg?react";

interface AddSkillModalProps {
  onClose: () => void;
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
        "mx-0.5 inline-block rounded-sm border border-[var(--oh-border-subtle)]",
        "bg-[var(--oh-surface-raised)] px-1.5 py-0.5 align-baseline font-mono text-[11px] text-white",
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
    await navigator.clipboard.writeText(ADD_SKILL_EXAMPLE_COMMAND);
    setCopied(true);
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
          "overflow-x-auto rounded-sm border border-[var(--oh-border-subtle)]",
          "bg-[var(--oh-surface-raised)] p-2 pr-10 text-xs text-white",
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
          "absolute right-2 top-2 cursor-pointer rounded-sm border border-[var(--oh-border-subtle)]",
          "bg-base-secondary p-1 text-tertiary-alt transition-colors",
          "hover:bg-[var(--oh-surface)] hover:text-white disabled:cursor-default [&_path]:fill-current",
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

export function AddSkillModal({ onClose }: AddSkillModalProps) {
  const { t } = useTranslation("openhands");

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_TITLE)}
    >
      <div
        data-testid="add-skill-modal"
        className="relative flex w-[520px] max-w-[90vw] max-h-[85vh] flex-col rounded-xl border border-[var(--oh-border)] bg-base-secondary"
      >
        <ModalCloseButton onClose={onClose} testId="add-skill-modal-close" />
        <header className="flex-shrink-0 px-6 pb-4 pt-6">
          <h2 className={cn("pr-6", modalTitleLgClassName)}>
            {t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_TITLE)}
          </h2>
          <p className="mt-4 text-sm text-tertiary-light">
            {t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_INTRO)}
          </p>
        </header>

        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-6 custom-scrollbar">
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
            className="self-start text-xs text-[var(--oh-muted)] transition-colors hover:text-white hover:underline"
          >
            {t(I18nKey.SETTINGS$SKILLS_ADD_MODAL_VIEW_DOCS)}
          </a>
        </div>

        <footer className="flex flex-shrink-0 justify-end gap-2 px-6 pb-6 pt-4">
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onClose}
            testId="add-skill-modal-dismiss"
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
