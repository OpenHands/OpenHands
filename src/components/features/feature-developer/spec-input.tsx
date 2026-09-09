import React from "react";
import { useTranslation } from "react-i18next";
import type { ProjectSummary } from "#/api/projects-service/projects-types";
import {
  DEFAULT_MAX_CONCURRENT_AGENTS,
  previewEstimateUsd,
} from "#/api/feature-developer-service/feature-developer-constants";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { CostText } from "#/components/shared/cost-text";
import { I18nKey } from "#/i18n/declaration";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";

export interface SpecInputProps {
  projects: ProjectSummary[];
  isPending?: boolean;
  onStart: (payload: {
    project_id: string;
    spec_text: string;
    max_concurrent_agents: number;
    continue_on_failure: boolean;
  }) => void;
}

export function SpecInput({ projects, isPending, onStart }: SpecInputProps) {
  const { t } = useTranslation("openhands");
  const [projectId, setProjectId] = React.useState("");
  const [spec, setSpec] = React.useState("");
  const [maxAgents, setMaxAgents] = React.useState(
    String(DEFAULT_MAX_CONCURRENT_AGENTS),
  );
  const [continueOnFailure, setContinueOnFailure] = React.useState(false);
  const [submitted, setSubmitted] = React.useState(false);

  const specError = submitted && !spec.trim();
  const projectError = submitted && !projectId;
  const estimate = previewEstimateUsd(spec);

  return (
    <form
      data-testid="feature-dev-composer"
      className="flex flex-col gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        setSubmitted(true);
        if (!spec.trim() || !projectId) {
          return;
        }
        onStart({
          project_id: projectId,
          spec_text: spec.trim(),
          max_concurrent_agents:
            Number(maxAgents) || DEFAULT_MAX_CONCURRENT_AGENTS,
          continue_on_failure: continueOnFailure,
        });
      }}
    >
      <SettingsDropdownInput
        testId="feature-dev-project"
        name="feature-dev-project"
        label={t(I18nKey.FEATURE_DEV$PROJECT)}
        placeholder={t(I18nKey.FEATURE_DEV$PROJECT_PLACEHOLDER)}
        selectedKey={projectId}
        items={projects.map((project) => ({
          key: project.id,
          label: project.name,
        }))}
        onSelectionChange={(key) => setProjectId(key ? String(key) : "")}
      />
      {projectError ? (
        <p
          data-testid="feature-dev-project-error"
          className="text-sm text-red-400"
        >
          {t(I18nKey.FEATURE_DEV$VALIDATION_PROJECT)}
        </p>
      ) : null}

      <label className="flex flex-col gap-2.5">
        <span className="text-sm">{t(I18nKey.FEATURE_DEV$SPEC)}</span>
        <textarea
          data-testid="feature-dev-spec"
          name="feature-dev-spec"
          value={spec}
          onChange={(event) => setSpec(event.target.value)}
          placeholder={t(I18nKey.FEATURE_DEV$SPEC_PLACEHOLDER)}
          className={`${formControlMultilineFieldClassName} min-h-32`}
        />
      </label>
      {specError ? (
        <p
          data-testid="feature-dev-spec-error"
          className="text-sm text-red-400"
        >
          {t(I18nKey.FEATURE_DEV$VALIDATION_SPEC)}
        </p>
      ) : null}

      <SettingsInput
        testId="feature-dev-max-agents"
        name="feature-dev-max-agents"
        type="number"
        min={1}
        label={t(I18nKey.FEATURE_DEV$MAX_AGENTS)}
        value={maxAgents}
        onChange={setMaxAgents}
      />

      <label className="flex items-center gap-2 text-sm text-white">
        <input
          data-testid="feature-dev-continue-on-failure"
          type="checkbox"
          checked={continueOnFailure}
          onChange={(event) => setContinueOnFailure(event.target.checked)}
        />
        {t(I18nKey.FEATURE_DEV$CONTINUE_ON_FAILURE)}
      </label>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <p
          data-testid="feature-dev-estimate"
          className="text-sm text-tertiary-light"
        >
          {t(I18nKey.FEATURE_DEV$ESTIMATE_PREVIEW)}
          <span className="ml-2 tabular-nums text-white">
            <CostText amount={estimate} />
          </span>
        </p>
        <BrandButton
          type="submit"
          variant="primary"
          testId="feature-dev-start"
          isDisabled={isPending}
        >
          {t(I18nKey.FEATURE_DEV$START)}
        </BrandButton>
      </div>
    </form>
  );
}
