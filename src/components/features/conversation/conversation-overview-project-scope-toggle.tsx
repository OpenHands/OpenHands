import { useTranslation } from "react-i18next";
import { SegmentedToggle } from "#/components/features/files-tab/segmented-toggle";
import { I18nKey } from "#/i18n/declaration";
import {
  CONVERSATION_OVERVIEW_PROJECT_SCOPE,
  type ConversationOverviewProjectScope,
} from "#/utils/conversation-overview-project-scope";

interface ConversationOverviewProjectScopeToggleProps {
  value: ConversationOverviewProjectScope;
  onChange: (value: ConversationOverviewProjectScope) => void;
  testId: string;
}

export function ConversationOverviewProjectScopeToggle({
  value,
  onChange,
  testId,
}: ConversationOverviewProjectScopeToggleProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex w-full justify-center [&>[role=radiogroup]]:w-full">
      <SegmentedToggle<ConversationOverviewProjectScope>
        value={value}
        onChange={onChange}
        ariaLabel={t(I18nKey.CONVERSATION$OVERVIEW_SCOPE_FILTER)}
        testId={testId}
        options={[
          {
            value: CONVERSATION_OVERVIEW_PROJECT_SCOPE.project,
            label: t(I18nKey.CONVERSATION$OVERVIEW_SCOPE_PROJECT),
          },
          {
            value: CONVERSATION_OVERVIEW_PROJECT_SCOPE.all,
            label: t(I18nKey.CONVERSATION$OVERVIEW_SCOPE_ALL),
          },
        ]}
      />
    </div>
  );
}
