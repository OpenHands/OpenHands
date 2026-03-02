import React from "react";
import { useTranslation } from "react-i18next";
import { useNavigate, redirect } from "react-router";
import OptionService from "#/api/option-service/option-service.api";
import { queryClient } from "#/query-client-config";
import StepHeader from "#/components/features/onboarding/step-header";
import { StepContent } from "#/components/features/onboarding/step-content";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import OpenHandsLogoWhite from "#/assets/branding/openhands-logo-white.svg?react";
import { useSubmitOnboarding } from "#/hooks/mutation/use-submit-onboarding";
import { useTracking } from "#/hooks/use-tracking";
import { ENABLE_ONBOARDING } from "#/utils/feature-flags";
import { cn } from "#/utils/utils";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";

export const clientLoader = async () => {
  const config = await queryClient.ensureQueryData({
    queryKey: ["config"],
    queryFn: OptionService.getConfig,
  });

  if (config.app_mode !== "saas" || !ENABLE_ONBOARDING()) {
    return redirect("/");
  }

  return null;
};

interface StepOption {
  id: string;
  labelKey?: I18nKey;
  label?: string;
}

interface FormStep {
  id: string;
  titleKey: I18nKey;
  options: StepOption[];
}

const steps: FormStep[] = [
  {
    id: "step1",
    titleKey: I18nKey.ONBOARDING$STEP1_TITLE,
    options: [
      {
        id: "software_engineer",
        labelKey: I18nKey.ONBOARDING$SOFTWARE_ENGINEER,
      },
      {
        id: "engineering_manager",
        labelKey: I18nKey.ONBOARDING$ENGINEERING_MANAGER,
      },
      {
        id: "cto_founder",
        labelKey: I18nKey.ONBOARDING$CTO_FOUNDER,
      },
      {
        id: "product_operations",
        labelKey: I18nKey.ONBOARDING$PRODUCT_OPERATIONS,
      },
      {
        id: "student_hobbyist",
        labelKey: I18nKey.ONBOARDING$STUDENT_HOBBYIST,
      },
      {
        id: "other",
        labelKey: I18nKey.ONBOARDING$OTHER,
      },
    ],
  },
  {
    id: "step2",
    titleKey: I18nKey.ONBOARDING$STEP2_TITLE,
    options: [
      {
        id: "solo",
        labelKey: I18nKey.ONBOARDING$SOLO,
      },
      {
        id: "org_2_10",
        labelKey: I18nKey.ONBOARDING$ORG_2_10,
      },
      {
        id: "org_11_50",
        labelKey: I18nKey.ONBOARDING$ORG_11_50,
      },
      {
        id: "org_51_200",
        labelKey: I18nKey.ONBOARDING$ORG_51_200,
      },
      {
        id: "org_200_1000",
        labelKey: I18nKey.ONBOARDING$ORG_200_1000,
      },
      {
        id: "org_1000_plus",
        labelKey: I18nKey.ONBOARDING$ORG_1000_PLUS,
      },
    ],
  },
  {
    id: "step3",
    titleKey: I18nKey.ONBOARDING$STEP3_TITLE,
    options: [
      {
        id: "new_features",
        labelKey: I18nKey.ONBOARDING$NEW_FEATURES,
      },
      {
        id: "app_from_scratch",
        labelKey: I18nKey.ONBOARDING$APP_FROM_SCRATCH,
      },
      {
        id: "fixing_bugs",
        labelKey: I18nKey.ONBOARDING$FIXING_BUGS,
      },
      {
        id: "refactoring",
        labelKey: I18nKey.ONBOARDING$REFACTORING,
      },
      {
        id: "automating_tasks",
        labelKey: I18nKey.ONBOARDING$AUTOMATING_TASKS,
      },
      {
        id: "not_sure",
        labelKey: I18nKey.ONBOARDING$NOT_SURE,
      },
    ],
  },
];

function OnboardingForm() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { mutate: submitOnboarding } = useSubmitOnboarding();
  const { trackOnboardingCompleted } = useTracking();

  const [currentStepIndex, setCurrentStepIndex] = React.useState(0);
  const [selections, setSelections] = React.useState<Record<string, string>>(
    {},
  );

  const currentStep = steps[currentStepIndex];
  const isLastStep = currentStepIndex === steps.length - 1;
  const isFirstStep = currentStepIndex === 0;
  const currentSelection = selections[currentStep.id] || null;

  const handleSelectOption = (optionId: string) => {
    setSelections((prev) => ({
      ...prev,
      [currentStep.id]: optionId,
    }));
  };

  const handleNext = () => {
    if (isLastStep) {
      submitOnboarding({ selections });
      try {
        trackOnboardingCompleted({
          role: selections.step1,
          orgSize: selections.step2,
          useCase: selections.step3,
        });
      } catch (error) {
        console.error("Failed to track onboarding:", error);
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

  const translatedOptions = currentStep.options.map((option) => ({
    id: option.id,
    label: option.labelKey ? t(option.labelKey) : option.label!,
  }));

  return (
    <ModalBackdrop>
      <div
        data-testid="onboarding-form"
        className="w-[500px] max-w-[calc(100vw-2rem)] mx-auto p-4 sm:p-6 flex flex-col justify-center overflow-hidden"
      >
        <div className="flex flex-col items-center mb-4">
          <OpenHandsLogoWhite width={55} height={55} />
        </div>
        <StepHeader
          title={t(currentStep.titleKey)}
          currentStep={currentStepIndex + 1}
          totalSteps={steps.length}
        />
        <StepContent
          options={translatedOptions}
          selectedOptionId={currentSelection}
          onSelectOption={handleSelectOption}
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
            isDisabled={!currentSelection}
            className={cn(
              "px-4 sm:px-6 py-2.5 bg-white text-black hover:bg-white/90",
              isFirstStep ? "w-1/2" : "flex-1", // keep "Next" button to the right. Even if "Back" button is not rendered
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
    </ModalBackdrop>
  );
}

export default OnboardingForm;
