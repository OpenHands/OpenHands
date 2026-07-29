import React from "react";
import { Trans } from "react-i18next";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useModelStore } from "#/stores/model-store";
import { I18nKey } from "#/i18n/declaration";
import InfoCircleIcon from "#/icons/info-circle.svg?react";
import type { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import { GenericEventMessage } from "./generic-event-message";

interface ProfileRowProps {
  profile: ProfileInfo;
}

function ProfileRow({ profile }: ProfileRowProps) {
  const [expanded, setExpanded] = React.useState(false);

  return (
    <div className="border border-neutral-700 rounded-md overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        aria-label={`Toggle details for ${profile.name}`}
        className="w-full py-1.5 px-2 text-left flex items-center gap-2 hover:bg-neutral-700 transition-colors cursor-pointer"
      >
        <span className="text-neutral-300">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
        <span className="font-normal text-neutral-200 text-sm">
          {profile.name}
        </span>
      </button>
      {expanded && (
        <>
          <hr className="border-neutral-700" />
          <div className="px-3 py-2 text-xs text-neutral-300 font-mono whitespace-pre-wrap">
            {`model:    ${profile.model ?? "—"}\n` +
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
                <span className="inline-flex items-center gap-1.5">
                  <InfoCircleIcon
                    width={14}
                    height={14}
                    className="shrink-0 text-neutral-400"
                    aria-hidden
                  />
                  <Trans
                    i18nKey={I18nKey.MODEL$SWITCHED_TO_PROFILE}
                    values={{ name: entry.switchedTo }}
                    components={{
                      cmd: (
                        <span className="font-mono text-neutral-200 bg-neutral-800 px-1 rounded" />
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
              <span>
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
                <span className="text-neutral-300 text-sm px-2 py-1 block">
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
