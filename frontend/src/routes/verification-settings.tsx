import React from "react";
import {
  SdkSectionHeaderProps,
  SdkSectionPage,
} from "#/components/features/settings/sdk-settings/sdk-section-page";
import { SchemaField } from "#/components/features/settings/sdk-settings/schema-field";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import { useSettings } from "#/hooks/query/use-settings";
import { SettingsScope, SettingsSchema } from "#/types/settings";
import {
  buildInitialSettingsFormValues,
  buildSdkSettingsPayload,
  getVisibleSettingsSections,
  SettingsDirtyState,
  SettingsFormValues,
} from "#/utils/sdk-settings-schema";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { requireOrgDefaultsRedirect } from "#/utils/org/saas-redirect-to-org-defaults-guard";

const CRITIC_FIELD_KEYS = new Set([
  "verification.critic_enabled",
  "verification.enable_iterative_refinement",
]);

const filterCriticSchema = (
  schema: SettingsSchema | null | undefined,
): SettingsSchema | null => {
  if (!schema) return null;

  const sections = schema.sections
    .map((section) => ({
      ...section,
      fields: section.fields.filter((field) =>
        CRITIC_FIELD_KEYS.has(field.key),
      ),
    }))
    .filter((section) => section.fields.length > 0);

  if (sections.length === 0) return null;

  return { ...schema, sections };
};

function VerificationSettingsHeader({
  criticSchema,
  criticValues,
  isDisabled,
  onCriticChange,
  renderTopContent,
}: {
  criticSchema: SettingsSchema | null;
  criticValues: SettingsFormValues;
  isDisabled: boolean;
  onCriticChange: (key: string, value: string | boolean) => void;
  renderTopContent?: () => React.ReactNode;
}) {
  const visibleSections = React.useMemo(() => {
    if (!criticSchema) return [];
    return getVisibleSettingsSections(criticSchema, criticValues, "all");
  }, [criticSchema, criticValues]);

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
                value={criticValues[field.key]}
                isDisabled={isDisabled}
                onChange={(nextValue) => onCriticChange(field.key, nextValue)}
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
  const criticSchema = React.useMemo(
    () => filterCriticSchema(agentSchemaQuery.data),
    [agentSchemaQuery.data],
  );
  const [criticValues, setCriticValues] = React.useState<SettingsFormValues>(
    {},
  );
  const [criticDirty, setCriticDirty] = React.useState<SettingsDirtyState>({});

  React.useEffect(() => {
    if (!settings || !criticSchema) return;

    setCriticValues(
      buildInitialSettingsFormValues(settings, criticSchema, "agent_settings"),
    );
    setCriticDirty({});
  }, [settings, criticSchema]);

  const handleCriticChange = React.useCallback(
    (key: string, value: string | boolean) => {
      setCriticValues((prev) => ({ ...prev, [key]: value }));
      setCriticDirty((prev) => ({ ...prev, [key]: true }));
    },
    [],
  );

  const buildHeader = React.useCallback(
    ({ isDisabled }: SdkSectionHeaderProps) => (
      <VerificationSettingsHeader
        criticSchema={criticSchema}
        criticValues={criticValues}
        isDisabled={isDisabled}
        onCriticChange={handleCriticChange}
        renderTopContent={renderTopContent}
      />
    ),
    [criticSchema, criticValues, handleCriticChange, renderTopContent],
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

      if (criticSchema && Object.keys(criticDirty).length > 0) {
        payload.agent_settings_diff = buildSdkSettingsPayload(
          criticSchema,
          criticValues,
          criticDirty,
        );
      }

      return payload;
    },
    [criticDirty, criticSchema, criticValues],
  );

  return (
    <SdkSectionPage
      scope={scope}
      settingsSource="conversation_settings"
      sectionKeys={["verification"]}
      header={buildHeader}
      extraDirty={Object.keys(criticDirty).length > 0}
      buildPayload={buildPayload}
      onSaveSuccess={() => setCriticDirty({})}
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
