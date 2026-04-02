import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router";
import { I18nKey } from "#/i18n/declaration";
import { useTracking } from "#/hooks/use-tracking";
import { Card } from "#/ui/card";
import { Typography } from "#/ui/typography";
import {
  clearEnterpriseFormData,
  EnterpriseFormData,
} from "#/utils/local-storage";
import { cn } from "#/utils/utils";
import { FormInput } from "./form-input";
import OpenHandsLogoWhite from "#/assets/branding/openhands-logo-white.svg?react";
import CloudIcon from "#/icons/cloud-minimal.svg?react";
import StackedIcon from "#/icons/stacked.svg?react";

export type RequestType = "saas" | "self-hosted";

interface InformationRequestFormProps {
  requestType: RequestType;
  formData: EnterpriseFormData;
  onFormDataChange: (data: EnterpriseFormData) => void;
  onBack: () => void;
}

export function InformationRequestForm({
  requestType,
  formData,
  onFormDataChange,
  onBack,
}: InformationRequestFormProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { trackEnterpriseLeadFormSubmitted } = useTracking();
  const [hasAttemptedSubmit, setHasAttemptedSubmit] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (isSubmitting) return;
    setHasAttemptedSubmit(true);

    // Use native form validation to show browser error popups
    const form = e.currentTarget;
    if (!form.checkValidity()) {
      form.reportValidity();
      return;
    }

    setIsSubmitting(true);

    trackEnterpriseLeadFormSubmitted({
      requestType,
      name: formData.name.trim(),
      company: formData.company.trim(),
      email: formData.email.trim(),
      message: formData.message.trim(),
    });

    // Clear form data from localStorage and reset form state
    clearEnterpriseFormData(requestType);
    onFormDataChange({ name: "", company: "", email: "", message: "" });

    // Navigate to login page with state to show confirmation modal
    navigate("/login", { state: { showRequestSubmittedModal: true } });
  };

  const isSaas = requestType === "saas";

  const title = isSaas
    ? t(I18nKey.ENTERPRISE$FORM_SAAS_TITLE)
    : t(I18nKey.ENTERPRISE$FORM_SELF_HOSTED_TITLE);

  const subtitle = isSaas
    ? t(I18nKey.ENTERPRISE$FORM_SAAS_SUBTITLE)
    : t(I18nKey.ENTERPRISE$FORM_SELF_HOSTED_SUBTITLE);

  const cardTitle = isSaas
    ? t(I18nKey.ENTERPRISE$SAAS_TITLE)
    : t(I18nKey.ENTERPRISE$SELF_HOSTED_TITLE);

  const cardDescription = isSaas
    ? t(I18nKey.ENTERPRISE$SAAS_DESCRIPTION)
    : t(I18nKey.ENTERPRISE$SELF_HOSTED_DESCRIPTION);

  const messagePlaceholder = isSaas
    ? t(I18nKey.ENTERPRISE$FORM_MESSAGE_SAAS_PLACEHOLDER)
    : t(I18nKey.ENTERPRISE$FORM_MESSAGE_SELF_HOSTED_PLACEHOLDER);

  return (
    <div
      data-testid="information-request-form"
      className={cn("w-full max-w-[896px] flex flex-col items-center gap-8")}
    >
      {/* Header */}
      <div className={cn("w-full flex flex-col items-center gap-4")}>
        <OpenHandsLogoWhite width={56} height={56} />
        <div className={cn("text-center flex flex-col gap-2")}>
          <Typography.H1 className={cn("text-2xl font-semibold")}>
            {title}
          </Typography.H1>
          <Typography.Text className={cn("text-[#8C8C8C] leading-5")}>
            {subtitle}
          </Typography.Text>
        </div>
      </div>

      {/* Content: Form + Card */}
      <div className={cn("w-full flex flex-col md:flex-row gap-8")}>
        {/* Form */}
        <form
          onSubmit={handleSubmit}
          className={cn("flex-1 flex flex-col gap-4 w-full md:max-w-[544px]")}
        >
          <FormInput
            id="name"
            label={t(I18nKey.ENTERPRISE$FORM_NAME_LABEL)}
            value={formData.name}
            placeholder={t(I18nKey.ENTERPRISE$FORM_NAME_PLACEHOLDER)}
            required
            showError={hasAttemptedSubmit}
            onChange={(value) => onFormDataChange({ ...formData, name: value })}
          />

          <FormInput
            id="company"
            label={t(I18nKey.ENTERPRISE$FORM_COMPANY_LABEL)}
            value={formData.company}
            placeholder={t(I18nKey.ENTERPRISE$FORM_COMPANY_PLACEHOLDER)}
            required
            showError={hasAttemptedSubmit}
            onChange={(value) =>
              onFormDataChange({ ...formData, company: value })
            }
          />

          <FormInput
            id="email"
            label={t(I18nKey.ENTERPRISE$FORM_EMAIL_LABEL)}
            type="email"
            value={formData.email}
            placeholder={t(I18nKey.ENTERPRISE$FORM_EMAIL_PLACEHOLDER)}
            required
            showError={hasAttemptedSubmit}
            onChange={(value) =>
              onFormDataChange({ ...formData, email: value })
            }
          />

          <FormInput
            id="message"
            label={t(I18nKey.ENTERPRISE$FORM_MESSAGE_LABEL)}
            value={formData.message}
            placeholder={messagePlaceholder}
            rows={4}
            required
            showError={hasAttemptedSubmit}
            onChange={(value) =>
              onFormDataChange({ ...formData, message: value })
            }
          />

          {/* Buttons */}
          <div
            className={cn("flex gap-4 mt-4")}
            role="group"
            aria-label="Form actions"
          >
            <button
              type="button"
              onClick={onBack}
              aria-label={t(I18nKey.COMMON$BACK)}
              className={cn(
                "flex-1 px-6 py-2.5 text-sm text-center rounded",
                "bg-transparent text-white border border-[#242424]",
                "hover:bg-white hover:text-black transition-colors cursor-pointer",
              )}
            >
              {t(I18nKey.COMMON$BACK)}
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              aria-label={t(I18nKey.ENTERPRISE$FORM_SUBMIT)}
              className={cn(
                "flex-1 px-6 py-2.5 text-sm rounded",
                "bg-white text-black border border-white",
                "hover:bg-gray-100 transition-colors cursor-pointer",
                "disabled:opacity-50 disabled:cursor-not-allowed",
              )}
            >
              {isSubmitting
                ? t(I18nKey.ENTERPRISE$FORM_SUBMITTING)
                : t(I18nKey.ENTERPRISE$FORM_SUBMIT)}
            </button>
          </div>
        </form>

        {/* CTA Card */}
        <Card
          theme="dark"
          gradient="standard"
          className={cn("w-full md:w-80 flex-col p-6 gap-4")}
        >
          <div className={cn("w-10 h-10")}>
            {isSaas ? (
              <CloudIcon className={cn("w-10 h-10 text-[#8C8C8C]")} />
            ) : (
              <StackedIcon className={cn("w-10 h-10")} />
            )}
          </div>
          <Typography.H3
            className={cn("text-xl font-semibold leading-7 text-[#FAFAFA]")}
          >
            {cardTitle}
          </Typography.H3>
          <Typography.Text
            className={cn(
              "relative top-[0.5px] font-inter text-[#8C8C8C]",
              "font-400 text-14px leading-[22.75px] tracking-[0px]",
            )}
          >
            {cardDescription}
          </Typography.Text>
        </Card>
      </div>
    </div>
  );
}
