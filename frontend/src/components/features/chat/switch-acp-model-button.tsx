import React from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useConfig } from "#/hooks/query/use-config";
import { useSwitchAcpModel } from "#/hooks/mutation/use-switch-acp-model";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { ACP_SERVER_TAG } from "#/utils/agent-display-label";
import { ContextMenu } from "#/ui/context-menu";
import { ContextMenuListItem } from "../context-menu/context-menu-list-item";
import { Divider } from "#/ui/divider";
import { SettingsNavHeader } from "../settings/settings-nav-header";
import { ToolsContextMenuIconText } from "../controls/tools-context-menu-icon-text";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { extractErrorMessage } from "#/utils/extract-error-message";
import { I18nKey } from "#/i18n/declaration";
import { CONTEXT_MENU_ICON_TEXT_CLASSNAME } from "#/utils/constants";
import { cn } from "#/utils/utils";
import ChevronDownSmallIcon from "#/icons/chevron-down-small.svg?react";
import CheckIcon from "#/icons/checkmark.svg?react";
import SettingsIcon from "#/icons/settings.svg?react";
import type { ACPModelOption } from "#/api/option-service/option.types";

interface AcpModelMenuProps {
  models: ACPModelOption[];
  activeModelId: string | null;
  onSelect: (modelId: string) => void;
  onClose: () => void;
}

function AcpModelMenu({
  models,
  activeModelId,
  onSelect,
  onClose,
}: AcpModelMenuProps) {
  const { t } = useTranslation();
  const ref = useClickOutsideElement<HTMLUListElement>(onClose);

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const handleSelect = (e: React.MouseEvent<HTMLButtonElement>, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    onSelect(id);
    onClose();
  };

  return (
    <ContextMenu
      ref={ref}
      testId="switch-acp-model-context-menu"
      position="top"
      alignment="left"
      className="left-0 mb-2 bottom-full min-w-[260px] max-h-[60vh] overflow-y-auto"
    >
      <SettingsNavHeader
        text={I18nKey.SETTINGS$AVAILABLE_MODELS}
        className="px-2 pt-1 pb-1"
      />
      {models.map((model) => {
        const isActive = model.id === activeModelId;
        return (
          <ContextMenuListItem
            key={model.id}
            testId={`switch-acp-model-option-${model.id}`}
            onClick={(e) => handleSelect(e, model.id)}
            className="cursor-pointer p-0 h-auto hover:bg-transparent"
          >
            <div
              className={cn(
                "flex items-center justify-between gap-2 p-2 rounded",
                isActive ? "bg-[#5C5D62]" : "hover:bg-[#5C5D62]",
              )}
            >
              <span className="truncate text-sm">{model.label}</span>
              {isActive && (
                <CheckIcon width={14} height={14} className="shrink-0" />
              )}
            </div>
          </ContextMenuListItem>
        );
      })}
      <Divider />
      <Link
        to="/settings/agent"
        onClick={onClose}
        data-testid="switch-acp-model-open-settings"
        className={cn("block", CONTEXT_MENU_ICON_TEXT_CLASSNAME)}
      >
        <ToolsContextMenuIconText
          icon={<SettingsIcon width={16} height={16} />}
          text={t(I18nKey.MODEL$OPEN_SETTINGS)}
          className={CONTEXT_MENU_ICON_TEXT_CLASSNAME}
        />
      </Link>
    </ContextMenu>
  );
}

export function SwitchAcpModelButton() {
  const { t } = useTranslation();
  const [menuOpen, setMenuOpen] = React.useState(false);
  const { conversationId } = useConversationId();
  const { data: conversation } = useActiveConversation();
  const { data: config } = useConfig();
  const { mutate, isPending: isSwitchPending } = useSwitchAcpModel();
  const { mutate: saveSettings, isPending: isSavePending } = useSaveSettings();
  const isPending = isSwitchPending || isSavePending;

  const isAcp = conversation?.agent_kind === "acp";
  const acpServerKey = conversation?.tags?.[ACP_SERVER_TAG] ?? null;
  const provider = config?.acp_providers?.find((p) => p.key === acpServerKey);
  const availableModels = provider?.available_models ?? [];
  const currentModelId = conversation?.llm_model ?? null;

  // Live switching requires an active ACP session (first run() must have
  // completed).  execution_status is null until the first run() fires.
  // Before that, persist the choice via settings so the conversation picks
  // it up when it starts.
  const isSessionLive = conversation?.execution_status != null;

  if (!isAcp || availableModels.length === 0) return null;

  const currentLabel =
    availableModels.find((m) => m.id === currentModelId)?.label ??
    currentModelId ??
    t(I18nKey.LLM$SELECT_MODEL_PLACEHOLDER);

  const handleSelect = (modelId: string) => {
    if (modelId === currentModelId) return;
    if (isSessionLive) {
      mutate(
        { conversationId, model: modelId },
        {
          onError: (err) =>
            displayErrorToast(
              extractErrorMessage(
                err,
                t(I18nKey.MODEL$SWITCH_FAILED, { name: modelId }),
              ),
            ),
        },
      );
    } else {
      saveSettings(
        { agent_settings_diff: { acp_model: modelId } },
        {
          onError: (err) =>
            displayErrorToast(
              extractErrorMessage(
                err,
                t(I18nKey.MODEL$SWITCH_FAILED, { name: modelId }),
              ),
            ),
        },
      );
    }
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setMenuOpen((o) => !o);
        }}
        disabled={isPending}
        data-testid="switch-acp-model-button"
        title={currentModelId ?? undefined}
        aria-haspopup="menu"
        aria-expanded={menuOpen}
        className="flex items-center gap-1 border border-[#4B505F] rounded-[100px] transition-opacity cursor-pointer hover:opacity-80 disabled:opacity-50 disabled:cursor-not-allowed pl-2 max-w-[200px]"
      >
        <span className="text-white text-2.75 not-italic font-normal leading-5 truncate">
          {currentLabel}
        </span>
        <ChevronDownSmallIcon
          width={24}
          height={24}
          color="#ffffff"
          className="shrink-0"
        />
      </button>
      {menuOpen && (
        <AcpModelMenu
          models={availableModels}
          activeModelId={currentModelId}
          onSelect={handleSelect}
          onClose={() => setMenuOpen(false)}
        />
      )}
    </div>
  );
}
