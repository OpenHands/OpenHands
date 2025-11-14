import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { SettingsInput } from "../settings-input";
import { AzureDevOpsTokenHelpAnchor } from "./azure-devops-token-help-anchor";
import { KeyStatusIcon } from "../key-status-icon";

interface AzureDevOpsTokenInputProps {
  onChange: (value: string) => void;
  onAzureDevOpsHostChange: (value: string) => void;
  isAzureDevOpsTokenSet: boolean;
  name: string;
  azureDevOpsHostSet: string | null | undefined;
}

export function AzureDevOpsTokenInput({
  onChange,
  onAzureDevOpsHostChange,
  isAzureDevOpsTokenSet,
  name,
  azureDevOpsHostSet,
}: AzureDevOpsTokenInputProps) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-6">
      <SettingsInput
        testId={name}
        name={name}
        onChange={onChange}
        label={t(I18nKey.GIT$AZURE_DEVOPS_TOKEN)}
        type="password"
        className="w-full max-w-[680px]"
        placeholder={isAzureDevOpsTokenSet ? "<hidden>" : ""}
        startContent={
          isAzureDevOpsTokenSet && (
            <KeyStatusIcon
              testId="azure-devops-set-token-indicator"
              isSet={isAzureDevOpsTokenSet}
            />
          )
        }
      />

      <SettingsInput
        onChange={onAzureDevOpsHostChange || (() => {})}
        name="azure-devops-host-input"
        testId="azure-devops-host-input"
        label={t(I18nKey.GIT$AZURE_DEVOPS_HOST)}
        type="text"
        className="w-full max-w-[680px]"
        placeholder={t(I18nKey.GIT$AZURE_DEVOPS_HOST_PLACEHOLDER)}
        defaultValue={azureDevOpsHostSet || undefined}
        startContent={
          azureDevOpsHostSet &&
          azureDevOpsHostSet.trim() !== "" && (
            <KeyStatusIcon testId="azure-devops-set-host-indicator" isSet />
          )
        }
      />

      <AzureDevOpsTokenHelpAnchor />
    </div>
  );
}
