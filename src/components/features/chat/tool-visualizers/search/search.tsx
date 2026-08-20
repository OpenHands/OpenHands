import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { defineVisualizer } from "../define";
import { textFromContent } from "../text-content";
import { KeyValueGrid } from "../primitives/key-value-grid";

/**
 * Search visualizer for `grep` / `glob`. Both cards show the pattern / path /
 * include parameters; the observation card adds the match count and the list of
 * matching files (or an error / empty state).
 */
export const searchVisualizer = defineVisualizer({
  actionKinds: ["GrepAction", "GlobAction"],
  observationKinds: ["GrepObservation", "GlobObservation"],
  Body: function SearchBody({ action, observation }) {
    const { t } = useTranslation("openhands");
    const obs = observation?.observation;
    const act = action?.action;

    const pattern = obs?.pattern ?? act?.pattern ?? "";
    const path = obs?.search_path ?? act?.path ?? "";
    const include = obs
      ? "include_pattern" in obs
        ? obs.include_pattern
        : null
      : act && "include" in act
        ? act.include
        : null;

    const results = obs ? ("matches" in obs ? obs.matches : obs.files) : [];

    const rows = [
      { label: t(I18nKey.COMMON$PATTERN), value: pattern },
      ...(path ? [{ label: t(I18nKey.COMMON$PATH), value: path }] : []),
      ...(include
        ? [{ label: t(I18nKey.COMMON$INCLUDE), value: include }]
        : []),
    ];

    return (
      <div className="flex flex-col gap-2">
        <KeyValueGrid rows={rows} />
        {obs &&
          (obs.is_error ? (
            <span className="whitespace-pre-wrap text-xs text-danger">
              {textFromContent(obs.content)}
            </span>
          ) : results.length === 0 ? (
            <span className="text-xs text-muted">
              {t(I18nKey.COMMON$NO_RESULTS)}
            </span>
          ) : (
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted">
                {t(I18nKey.COMMON$RESULTS, { count: results.length })}
              </span>
              <div className="flex flex-col gap-0.5 font-mono text-xs text-foreground">
                {results.map((file) => (
                  <span key={file} className="break-all">
                    {file}
                  </span>
                ))}
              </div>
              {obs.truncated && (
                <span className="text-xs text-muted">
                  {t(I18nKey.COMMON$TRUNCATED)}
                </span>
              )}
            </div>
          ))}
      </div>
    );
  },
});
