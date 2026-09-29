import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";
import {
  PENDING_AUTOMATION_SETUP_ID,
  getAutomationFormSession,
  subscribeAutomationFormSession,
} from "#/api/automation-form-session";
import type { AutomationSetupDraft } from "#/api/automation-setup-types";
import { InteractiveChatBox } from "#/components/features/chat/interactive-chat-box";
import { AutomationSetupPanel } from "#/components/features/automations/setup/automation-setup-panel";
import { ComposerDockedProvider } from "#/context/composer-docked-context";
import { useStartAutomationSetup } from "#/hooks/use-start-automation-setup";
import { I18nKey } from "#/i18n/declaration";

const BLANK_DRAFT: AutomationSetupDraft = { prompt: "", kind: "prompt" };

/**
 * New automation form before any conversation exists.
 *
 * The agent drawer stays hidden. Sending the docked prompt is what creates
 * the conversation and reveals the agent.
 */
export default function AutomationSetupNew() {
  const { t } = useTranslation("openhands");
  const { startConversationFromPrompt, isPending } = useStartAutomationSetup();
  const [toolbarElement, setToolbarElement] = useState<HTMLDivElement | null>(
    null,
  );
  const [composerTarget, setComposerTarget] = useState<HTMLDivElement | null>(
    null,
  );
  const [draft, setDraft] = useState<AutomationSetupDraft>(
    () => getAutomationFormSession(PENDING_AUTOMATION_SETUP_ID) ?? BLANK_DRAFT,
  );

  useEffect(
    () =>
      subscribeAutomationFormSession(PENDING_AUTOMATION_SETUP_ID, (next) => {
        setDraft(next ?? BLANK_DRAFT);
      }),
    [],
  );

  const title = draft.form?.name?.trim() || t(I18nKey.AUTOMATION_SETUP$TITLE);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-base">
      <header
        data-testid="automation-setup-topbar"
        className="flex h-10 min-h-10 shrink-0 items-center justify-between gap-2 border-b border-[var(--oh-border)] bg-base px-3"
      >
        <h2 className="min-w-0 truncate text-sm font-medium text-content">
          {title}
        </h2>
        <div
          ref={setToolbarElement}
          data-testid="automation-setup-toolbar"
          className="flex min-w-0 shrink-0 items-center gap-1.5"
        />
      </header>
      <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
        <AutomationSetupPanel
          draft={draft}
          conversationId={PENDING_AUTOMATION_SETUP_ID}
          toolbarPortal={toolbarElement}
          showInlineHeader={false}
          reserveComposerSpace
        />
        <div className="custom-scrollbar-always pointer-events-none absolute inset-0 z-20 overflow-y-scroll px-5 [scrollbar-gutter:stable] [&::-webkit-scrollbar-thumb]:bg-transparent">
          <div className="relative mx-auto h-full w-full min-w-0 max-w-[800px]">
            <div
              ref={setComposerTarget}
              data-testid="automation-setup-docked-composer"
              className="pointer-events-auto absolute inset-x-0 bottom-5 overflow-visible rounded-[15px] shadow-[0_12px_40px_rgba(0,0,0,0.55)]"
            />
          </div>
        </div>
        {composerTarget
          ? createPortal(
              <ComposerDockedProvider enabled minimal>
                <InteractiveChatBox
                  onSubmit={(content) => startConversationFromPrompt(content)}
                  disabled={isPending}
                  showGitControlBar={false}
                  placeholder={t(
                    I18nKey.AUTOMATION_SETUP$DOCKED_COMPOSER_PLACEHOLDER,
                  )}
                />
              </ComposerDockedProvider>,
              composerTarget,
            )
          : null}
      </div>
    </div>
  );
}
