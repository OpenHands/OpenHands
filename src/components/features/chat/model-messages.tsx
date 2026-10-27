import React from "react";
import { Trans } from "react-i18next";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useModelStore } from "#/stores/model-store";
import { I18nKey } from "#/i18n/declaration";
import InfoCircleIcon from "#/icons/info-circle.svg?react";
import type { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import { useFreeModels } from "#/hooks/query/use-free-models";
import { formatModelNameForDisplay } from "#/utils/format-model-name";
import { GenericEventMessage } from "./generic-event-message";

interface ProfileRowProps {
  profile: ProfileInfo;
}

function ProfileRow({ profile }: ProfileRowProps) {
  const [expanded, setExpanded] = React.useState(false);
  const freeModels = useFreeModels();
  const displayModel = formatModelNameForDisplay(profile.model, freeModels);

  return (
    <div className="border border-border rounded-md overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        aria-label={`Toggle details for ${profile.name}`}
        className="w-full py-1.5 px-2 text-left flex items-center gap-2 hover:bg-interactive-hover-low transition-colors cursor-pointer"
      >
        <span className="text-text-tertiary">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
        <span className="font-normal text-contrast text-sm">
          {profile.name}
        </span>
      </button>
      {expanded && (
        <>
          <hr className="border-border" />
          <div className="px-3 py-2 text-xs text-text-tertiary font-mono whitespace-pre-wrap">
            {`model:    ${displayModel ?? "—"}\n` +
              `base_url: ${profile.base_url ?? "—"}\n` +
              // eslint-disable-next-line i18next/no-literal-string -- diagnostic readout; field labels are intentionally untranslated
              `api_key:  ${profile.api_key_set ? "set" : "not set"}`}
          </div>
        </>
      )}
    </div>
  );
}

export interface ModelMessagesProps {
  conversationId: string | null | undefined;
  /**
   * Render only entries anchored to this event id. Use `null` to render the
   * "no events at the time of /model" entries (top of the chat history).
   */
  anchorEventId: string | null;
}

export function ModelMessages({
  conversationId,
  anchorEventId,
}: ModelMessagesProps) {
  const entriesById = useModelStore((s) => s.entriesByConversation);
  const allEntries = conversationId ? (entriesById[conversationId] ?? []) : [];
  const entries = allEntries.filter((e) => e.anchorEventId === anchorEventId);

  if (!conversationId || entries.length === 0) return null;

  return (
    <div data-testid="model-messages" className="flex flex-col w-full">
      {entries.map((entry) => {
        if (entry.switchedTo) {
          return (
            <GenericEventMessage
              key={entry.id}
              title={
                <span className="inline-flex items-center gap-1.5 text-contrast">
                  <InfoCircleIcon
                    width={14}
                    height={14}
                    className="shrink-0 text-text-tertiary"
                    aria-hidden
                  />
                  <Trans
                    i18nKey={I18nKey.MODEL$SWITCHED_TO_PROFILE}
                    values={{ name: entry.switchedTo }}
                    components={{
                      cmd: (
                        <span className="font-mono text-contrast bg-surface-raised px-1 rounded" />
                      ),
                    }}
                  />
                </span>
              }
              details=""
            />
          );
        }

        const isEmpty = entry.profiles.length === 0;
        return (
          <GenericEventMessage
            key={entry.id}
            title={
              <span className="text-contrast">
                {isEmpty ? (
                  <Trans i18nKey={I18nKey.MODEL$NO_SAVED_PROFILES} />
                ) : (
                  <Trans
                    i18nKey={I18nKey.MODEL$AVAILABLE_PROFILES}
                    values={{ count: entry.profiles.length }}
                  />
                )}
              </span>
            }
            details={
              isEmpty ? (
                <span className="text-text-tertiary text-sm px-2 py-1 block">
                  <Trans i18nKey={I18nKey.MODEL$NO_PROFILES_HINT} />
                </span>
              ) : (
                <div className="flex flex-col gap-1 mt-1">
                  {entry.profiles.map((p) => (
                    <ProfileRow key={p.name} profile={p} />
                  ))}
                </div>
              )
            }
            initiallyExpanded={isEmpty}
          />
        );
      })}
    </div>
  );
}
