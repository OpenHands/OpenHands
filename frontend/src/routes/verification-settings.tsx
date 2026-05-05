import React from "react";
import {
  SdkSectionHeaderProps,
  SdkSectionPage,
} from "#/components/features/settings/sdk-settings/sdk-section-page";
import { SchemaField } from "#/components/features/settings/sdk-settings/schema-field";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import { useSettings } from "#/hooks/query/use-settings";
import {
  SettingsFieldSchema,
  SettingsScope,
  SettingsSchema,
} from "#/types/settings";
import {
  buildInitialSettingsFormValues,
  buildSdkSettingsPayload,
  isSettingsFieldVisible,
  SettingsDirtyState,
  SettingsFormValues,
  SettingsView,
} from "#/utils/sdk-settings-schema";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { requireOrgDefaultsRedirect } from "#/utils/org/saas-redirect-to-org-defaults-guard";

const BASIC_AGENT_VERIFICATION_FIELD_KEYS = new Set([
  "verification.critic_enabled",
  "verification.enable_iterative_refinement",
]);

const CONVERSATION_OWNED_AGENT_VERIFICATION_FIELD_KEYS = new Set([
  "verification.confirmation_mode",
  "verification.security_analyzer",
]);

const getAgentVerificationSchema = (
  schema: SettingsSchema | null | undefined,
): SettingsSchema | null => {
  if (!schema) return null;

  const sections = schema.sections
    .filter((section) => section.key === "verification")
    .map((section) => ({
      ...section,
      fields: section.fields.filter(
        (field) =>
          !CONVERSATION_OWNED_AGENT_VERIFICATION_FIELD_KEYS.has(field.key),
      ),
    }))
    .filter((section) => section.fields.length > 0);

  if (sections.length === 0) return null;

  return { ...schema, sections };
};

const shouldShowAgentVerificationField = (
  field: SettingsFieldSchema,
  values: SettingsFormValues,
  view: SettingsView,
) => {
  if (!isSettingsFieldVisible(field, values)) return false;
  if (view === "basic") {
    return BASIC_AGENT_VERIFICATION_FIELD_KEYS.has(field.key);
  }
  return true;
};

function VerificationSettingsHeader({
  agentVerificationSchema,
  agentVerificationValues,
  isDisabled,
  view,
  onAgentVerificationChange,
  renderTopContent,
}: {
  agentVerificationSchema: SettingsSchema | null;
  agentVerificationValues: SettingsFormValues;
  isDisabled: boolean;
  view: SettingsView;
  onAgentVerificationChange: (key: string, value: string | boolean) => void;
  renderTopContent?: () => React.ReactNode;
}) {
  const visibleSections = React.useMemo(() => {
    if (!agentVerificationSchema) return [];
    return agentVerificationSchema.sections
      .map((section) => ({
        ...section,
        fields: section.fields.filter((field) =>
          shouldShowAgentVerificationField(
            field,
            agentVerificationValues,
            view,
          ),
        ),
      }))
      .filter((section) => section.fields.length > 0);
  }, [agentVerificationSchema, agentVerificationValues, view]);

  return (
    <div className="flex flex-col gap-6">
      {renderTopContent?.()}

      {visibleSections.map((section) => (
        <section key={section.key} className="flex flex-col gap-4">
          <div className="grid gap-4 xl:grid-cols-2">
            {section.fields.map((field) => (
              <SchemaField
                key={field.key}
                field={field}
                value={agentVerificationValues[field.key]}
                isDisabled={isDisabled}
                onChange={(nextValue) =>
                  onAgentVerificationChange(field.key, nextValue)
                }
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

export function VerificationSettingsScreen({
  scope = "personal",
  renderTopContent,
  testId = "verification-settings-screen",
}: {
  scope?: SettingsScope;
  renderTopContent?: () => React.ReactNode;
  testId?: string;
}) {
  const { data: settings } = useSettings(scope);
  const agentSchemaQuery = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );
  const agentVerificationSchema = React.useMemo(
    () => getAgentVerificationSchema(agentSchemaQuery.data),
    [agentSchemaQuery.data],
  );
  const [agentVerificationValues, setAgentVerificationValues] =
    React.useState<SettingsFormValues>({});
  const [agentVerificationDirty, setAgentVerificationDirty] =
    React.useState<SettingsDirtyState>({});

  React.useEffect(() => {
    if (!settings || !agentVerificationSchema) return;

    setAgentVerificationValues(
      buildInitialSettingsFormValues(
        settings,
        agentVerificationSchema,
        "agent_settings",
      ),
    );
    setAgentVerificationDirty({});
  }, [settings, agentVerificationSchema]);

  const handleAgentVerificationChange = React.useCallback(
    (key: string, value: string | boolean) => {
      setAgentVerificationValues((prev) => ({ ...prev, [key]: value }));
      setAgentVerificationDirty((prev) => ({ ...prev, [key]: true }));
    },
    [],
  );

  const buildHeader = React.useCallback(
    ({ isDisabled, view }: SdkSectionHeaderProps) => (
      <VerificationSettingsHeader
        agentVerificationSchema={agentVerificationSchema}
        agentVerificationValues={agentVerificationValues}
        isDisabled={isDisabled}
        view={view}
        onAgentVerificationChange={handleAgentVerificationChange}
        renderTopContent={renderTopContent}
      />
    ),
    [
      agentVerificationSchema,
      agentVerificationValues,
      handleAgentVerificationChange,
      renderTopContent,
    ],
  );

  const buildPayload = React.useCallback(
    (
      conversationPayload: Record<string, unknown>,
      {
        dirty,
      }: {
        dirty: SettingsDirtyState;
      },
    ) => {
      const payload: Record<string, unknown> = {};

      if (
        Object.keys(dirty).length > 0 &&
        Object.keys(conversationPayload).length > 0
      ) {
        payload.conversation_settings_diff = conversationPayload;
      }

      if (
        agentVerificationSchema &&
        Object.keys(agentVerificationDirty).length > 0
      ) {
        payload.agent_settings_diff = buildSdkSettingsPayload(
          agentVerificationSchema,
          agentVerificationValues,
          agentVerificationDirty,
        );
      }

      return payload;
    },
    [agentVerificationDirty, agentVerificationSchema, agentVerificationValues],
  );

  return (
    <SdkSectionPage
      scope={scope}
      settingsSource="conversation_settings"
      sectionKeys={["verification"]}
      header={buildHeader}
      extraDirty={Object.keys(agentVerificationDirty).length > 0}
      buildPayload={buildPayload}
      onSaveSuccess={() => setAgentVerificationDirty({})}
      testId={testId}
    />
  );
}

const orgDefaultsRedirectGuard = requireOrgDefaultsRedirect(
  "/settings/org-defaults/verification",
);
const verificationPermissionGuard = createPermissionGuard("view_llm_settings");

export const clientLoader = async (args: { request: Request }) => {
  const blocked = await orgDefaultsRedirectGuard(args);
  if (blocked) return blocked;
  return verificationPermissionGuard(args);
};

export default VerificationSettingsScreen;
