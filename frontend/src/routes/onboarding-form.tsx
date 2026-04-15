import React from "react";
import { useTranslation } from "react-i18next";
import { useNavigate, redirect } from "react-router";
import StepHeader from "#/components/features/onboarding/step-header";
import { StepContent } from "#/components/features/onboarding/step-content";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import OpenHandsLogoWhite from "#/assets/branding/openhands-logo-white.svg?react";
import { useSubmitOnboarding } from "#/hooks/mutation/use-submit-onboarding";
import { useTracking } from "#/hooks/use-tracking";
import { ENABLE_ONBOARDING } from "#/utils/feature-flags";
import { cn } from "#/utils/utils";
import { useConfig } from "#/hooks/query/use-config";
import {
  ONBOARDING_FORM,
  OnboardingQuestion,
  OnboardingAppMode,
} from "#/constants/onboarding";

export const clientLoader = async () => {
  if (!ENABLE_ONBOARDING()) {
    return redirect("/");
  }

  return null;
};

type OnboardingAnswers = Record<string, string | string[]>;

function getOnboardingAppMode(): OnboardingAppMode {
  // TODO: query for app mode (saas or self hosted super user)
  return "saas";
}

function getAnswerAsArray(answers: OnboardingAnswers, key: string): string[] {
  const value = answers[key];
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

function getTranslatedOptions(
  step: OnboardingQuestion,
  t: (key: I18nKey) => string,
) {
  if (step.type === "input") return undefined;
  return step.answerOptions.map((option) => ({
    id: option.id,
    label: t(option.key),
  }));
}

function getTranslatedInputFields(
  step: OnboardingQuestion,
  t: (key: I18nKey) => string,
) {
  if (step.type !== "input") return undefined;
  return step.inputOptions.map((field) => ({
    id: field.id,
    label: t(field.key),
  }));
}

function OnboardingForm() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const config = useConfig({ enabled: true });
  const { mutate: submitOnboarding } = useSubmitOnboarding();
  const { trackOnboardingCompleted } = useTracking();

  const onboardingAppMode: OnboardingAppMode = getOnboardingAppMode();

  const steps = React.useMemo(
    () =>
      ONBOARDING_FORM.filter((step) =>
        step.app_mode.includes(onboardingAppMode),
      ),
    [onboardingAppMode],
  );

  const [currentStepIndex, setCurrentStepIndex] = React.useState(0);
  const [answers, setAnswers] = React.useState<OnboardingAnswers>({});

  const currentStep = steps[currentStepIndex];
  const isLastStep = currentStepIndex === steps.length - 1;
  const isFirstStep = currentStepIndex === 0;

  const currentSelections = React.useMemo(
    () => (currentStep ? getAnswerAsArray(answers, currentStep.id) : []),
    [answers, currentStep],
  );

  const isStepComplete = React.useMemo(() => {
    if (!currentStep) return false;

    if (currentStep.type === "input") {
      return currentStep.inputOptions.every((field) => {
        const value = answers[field.id];
        return typeof value === "string" && value.trim() !== "";
      });
    }
    return currentSelections.length > 0;
  }, [currentStep, answers, currentSelections]);

  const inputValues = React.useMemo(() => {
    const result: Record<string, string> = {};
    for (const [key, value] of Object.entries(answers)) {
      if (typeof value === "string") {
        result[key] = value;
      }
    }
    return result;
  }, [answers]);

  const handleSelectOption = (optionId: string) => {
    if (!currentStep) return;

    if (currentStep.type === "multi") {
      setAnswers((prev) => {
        const currentArray = getAnswerAsArray(prev, currentStep.id);

        if (currentArray.includes(optionId)) {
          return {
            ...prev,
            [currentStep.id]: currentArray.filter((id) => id !== optionId),
          };
        }
        return {
          ...prev,
          [currentStep.id]: [...currentArray, optionId],
        };
      });
    } else {
      setAnswers((prev) => ({
        ...prev,
        [currentStep.id]: optionId,
      }));
    }
  };

  const handleInputChange = (fieldId: string, value: string) => {
    setAnswers((prev) => ({
      ...prev,
      [fieldId]: value,
    }));
  };

  const handleNext = () => {
    if (isLastStep) {
      submitOnboarding({ selections: answers });

      // Only track onboarding for SaaS users
      if (config.data?.app_mode === "saas") {
        trackOnboardingCompleted({
          role: typeof answers.role === "string" ? answers.role : undefined,
          orgSize:
            typeof answers.org_size === "string" ? answers.org_size : undefined,
          useCase: Array.isArray(answers.use_case)
            ? answers.use_case
            : undefined,
        });
      }
    } else {
      setCurrentStepIndex((prev) => prev + 1);
    }
  };

  const handleBack = () => {
    if (isFirstStep) {
      navigate(-1);
    } else {
      setCurrentStepIndex((prev) => prev - 1);
    }
  };

  if (!currentStep) {
    return null;
  }

  const translatedOptions = getTranslatedOptions(currentStep, t);
  const translatedInputFields = getTranslatedInputFields(currentStep, t);

  return (
    <div
      data-testid="onboarding-form"
      className="w-[500px] max-w-[calc(100vw-2rem)] mx-auto p-4 sm:p-6 flex flex-col justify-center overflow-hidden"
    >
      <div className="flex flex-col items-center mb-4">
        <OpenHandsLogoWhite width={55} height={55} />
      </div>
      <StepHeader
        title={t(currentStep.questionKey)}
        subtitle={
          currentStep.subtitleKey ? t(currentStep.subtitleKey) : undefined
        }
        currentStep={currentStepIndex + 1}
        totalSteps={steps.length}
      />
      <StepContent
        options={translatedOptions}
        inputFields={translatedInputFields}
        selectedOptionIds={currentSelections}
        inputValues={inputValues}
        onSelectOption={handleSelectOption}
        onInputChange={handleInputChange}
      />
      <div
        data-testid="step-actions"
        className="flex justify-end items-center gap-3"
      >
        {!isFirstStep && (
          <BrandButton
            type="button"
            variant="secondary"
            onClick={handleBack}
            className="flex-1 px-4 sm:px-6 py-2.5 bg-[050505] text-white border hover:bg-white border-[#242424] hover:text-black"
          >
            {t(I18nKey.ONBOARDING$BACK_BUTTON)}
          </BrandButton>
        )}
        <BrandButton
          type="button"
          variant="primary"
          onClick={handleNext}
          isDisabled={!isStepComplete}
          className={cn(
            "px-4 sm:px-6 py-2.5 bg-white text-black hover:bg-white/90",
            isFirstStep ? "w-1/2" : "flex-1",
          )}
        >
          {t(
            isLastStep
              ? I18nKey.ONBOARDING$FINISH_BUTTON
              : I18nKey.ONBOARDING$NEXT_BUTTON,
          )}
        </BrandButton>
      </div>
    </div>
  );
}

export default OnboardingForm;
