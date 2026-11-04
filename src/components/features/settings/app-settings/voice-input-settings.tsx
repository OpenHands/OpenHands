import React from "react";
import { useTranslation } from "react-i18next";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import {
  DEFAULT_TRANSCRIPTION_MODEL,
  readTranscriptionEndpoint,
  writeTranscriptionEndpoint,
} from "#/utils/transcription-endpoint-storage";

const TRANSCRIPTION_BASE_URL_PLACEHOLDER = "https://api.openai.com/v1";

/** Saved on change to browser storage, independently of the settings form. */
export function VoiceInputSettings() {
  const { t } = useTranslation("openhands");
  const [endpoint] = React.useState(readTranscriptionEndpoint);

  return (
    <div className="border-t border-border pt-6 mt-2">
      <h3 className="text-lg font-medium mb-2">
        {t(I18nKey.SETTINGS$VOICE_INPUT)}
      </h3>
      <p className="mb-4 text-sm leading-5 text-tertiary-light">
        {t(I18nKey.SETTINGS$VOICE_INPUT_DESCRIPTION)}
      </p>
      <div className="flex flex-col gap-6">
        <SettingsInput
          testId="transcription-base-url-input"
          type="text"
          label={t(I18nKey.SETTINGS$BASE_URL)}
          defaultValue={endpoint.baseUrl}
          placeholder={TRANSCRIPTION_BASE_URL_PLACEHOLDER}
          onChange={(baseUrl) => writeTranscriptionEndpoint({ baseUrl })}
          showOptionalTag
          className="w-full min-w-0"
        />
        <SettingsInput
          testId="transcription-api-key-input"
          type="password"
          label={t(I18nKey.SETTINGS_FORM$API_KEY)}
          defaultValue={endpoint.apiKey}
          onChange={(apiKey) => writeTranscriptionEndpoint({ apiKey })}
          showOptionalTag
          className="w-full min-w-0"
        />
        <SettingsInput
          testId="transcription-model-input"
          type="text"
          label={t(I18nKey.SETTINGS$AGENT_MODEL)}
          defaultValue={endpoint.model}
          placeholder={DEFAULT_TRANSCRIPTION_MODEL}
          onChange={(model) => writeTranscriptionEndpoint({ model })}
          showOptionalTag
          className="w-full min-w-0"
        />
      </div>
    </div>
  );
}
