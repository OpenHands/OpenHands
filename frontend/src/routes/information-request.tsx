import { useState, useEffect, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import {
  InformationRequestForm,
  RequestType,
} from "#/components/features/onboarding/information-request-form";
import { EnterpriseCard } from "#/components/features/onboarding/enterprise-card";
import OpenHandsLogoWhite from "#/assets/branding/openhands-logo-white.svg?react";
import CloudIcon from "#/icons/cloud-minimal.svg?react";
import StackedIcon from "#/icons/stacked.svg?react";
import {
  EnterpriseFormData,
  getEnterpriseFormData,
  saveEnterpriseFormData,
} from "#/utils/local-storage";
import { cn } from "#/utils/utils";

const DEFAULT_FORM_DATA: EnterpriseFormData = {
  name: "",
  company: "",
  email: "",
  message: "",
};

export default function InformationRequest() {
  const { t } = useTranslation();
  const [selectedRequestType, setSelectedRequestType] =
    useState<RequestType | null>(null);
  const [saasFormData, setSaasFormData] =
    useState<EnterpriseFormData>(DEFAULT_FORM_DATA);
  const [selfHostedFormData, setSelfHostedFormData] =
    useState<EnterpriseFormData>(DEFAULT_FORM_DATA);

  // Load saved form data from localStorage on mount
  useEffect(() => {
    const savedSaasData = getEnterpriseFormData("saas");
    if (savedSaasData) {
      setSaasFormData(savedSaasData);
    }

    const savedSelfHostedData = getEnterpriseFormData("self-hosted");
    if (savedSelfHostedData) {
      setSelfHostedFormData(savedSelfHostedData);
    }
  }, []);

  const handleLearnMore = (type: RequestType) => {
    setSelectedRequestType(type);
  };

  const handleFormBack = () => {
    setSelectedRequestType(null);
  };

  const handleFormDataChange = useCallback(
    (data: EnterpriseFormData) => {
      if (selectedRequestType === "saas") {
        setSaasFormData(data);
        saveEnterpriseFormData("saas", data);
      } else if (selectedRequestType === "self-hosted") {
        setSelfHostedFormData(data);
        saveEnterpriseFormData("self-hosted", data);
      }
    },
    [selectedRequestType],
  );

  const currentFormData =
    selectedRequestType === "saas" ? saasFormData : selfHostedFormData;

  const saasFeatures = [
    t(I18nKey.ENTERPRISE$SAAS_FEATURE_NO_INFRASTRUCTURE),
    t(I18nKey.ENTERPRISE$SAAS_FEATURE_SSO),
    t(I18nKey.ENTERPRISE$SAAS_FEATURE_ACCESS_ANYWHERE),
    t(I18nKey.ENTERPRISE$SAAS_FEATURE_AUTO_UPDATES),
  ];

  const selfHostedFeatures = [
    t(I18nKey.ENTERPRISE$SELF_HOSTED_FEATURE_ON_PREMISES),
    t(I18nKey.ENTERPRISE$SELF_HOSTED_FEATURE_DATA_CONTROL),
    t(I18nKey.ENTERPRISE$SELF_HOSTED_FEATURE_COMPLIANCE),
    t(I18nKey.ENTERPRISE$SELF_HOSTED_FEATURE_SUPPORT),
  ];

  // Show form if a request type is selected
  if (selectedRequestType) {
    return (
      <main
        data-testid="information-request-page"
        className={cn(
          "min-h-screen flex items-center justify-center bg-base p-4",
        )}
      >
        <div
          className={cn(
            "w-full max-w-4xl flex flex-col items-center gap-8 p-6",
          )}
        >
          <InformationRequestForm
            requestType={selectedRequestType}
            formData={currentFormData}
            onFormDataChange={handleFormDataChange}
            onBack={handleFormBack}
          />
        </div>
      </main>
    );
  }

  return (
    <main
      data-testid="information-request-page"
      className={cn(
        "min-h-screen flex items-center justify-center bg-base p-4",
      )}
    >
      <div
        className={cn(
          "w-full max-w-4xl flex flex-col items-center gap-[16px] p-6",
        )}
      >
        {/* Logo */}
        <OpenHandsLogoWhite width={56} height={56} />

        {/* Header */}
        <div className={cn("text-center flex flex-col gap-3")}>
          <Typography.H1 className={cn("text-2xl font-bold")}>
            {t(I18nKey.ENTERPRISE$GET_OPENHANDS_TITLE)}
          </Typography.H1>
          <Typography.Text className={cn("text-[#8C8C8C] max-w-lg")}>
            {t(I18nKey.ENTERPRISE$GET_OPENHANDS_SUBTITLE)}
          </Typography.Text>
        </div>

        {/* Cards */}
        <div className={cn("w-full flex flex-col md:flex-row gap-4")}>
          <EnterpriseCard
            icon={<CloudIcon className={cn("w-10 h-10 text-[#8C8C8C]")} />}
            title={t(I18nKey.ENTERPRISE$SAAS_TITLE)}
            description={t(I18nKey.ENTERPRISE$SAAS_DESCRIPTION)}
            features={saasFeatures}
            onLearnMore={() => handleLearnMore("saas")}
            learnMoreLabel={t(I18nKey.ENTERPRISE$LEARN_MORE)}
          />
          <EnterpriseCard
            icon={<StackedIcon className={cn("w-10 h-10")} />}
            title={t(I18nKey.ENTERPRISE$SELF_HOSTED_TITLE)}
            description={t(I18nKey.ENTERPRISE$SELF_HOSTED_CARD_DESCRIPTION)}
            features={selfHostedFeatures}
            onLearnMore={() => handleLearnMore("self-hosted")}
            learnMoreLabel={t(I18nKey.ENTERPRISE$LEARN_MORE)}
          />
        </div>

        {/* Back Link */}
        <Link
          to="/login"
          aria-label={t(I18nKey.COMMON$BACK)}
          className={cn(
            "px-6 py-2.5 text-sm rounded-sm",
            "bg-[#050505] text-white border border-[#242424]",
            "hover:bg-white hover:text-black transition-colors",
          )}
        >
          {t(I18nKey.COMMON$BACK)}
        </Link>
      </div>
    </main>
  );
}
