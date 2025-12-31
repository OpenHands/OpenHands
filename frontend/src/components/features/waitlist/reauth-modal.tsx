import React from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { I18nKey } from "#/i18n/declaration";
import OpenHandsLogo from "#/assets/branding/openhands-logo.svg?react";

export function ReauthModal() {
  const { t } = useTranslation();

  return (
    <ModalBackdrop>
      <ModalBody className="border border-tertiary">
        <OpenHandsLogo width={68} height={46} />
        <div className="flex flex-col gap-2 w-full items-center text-center">
          <h1 className="text-2xl font-bold">
            {t(I18nKey.AUTH$LOGGING_BACK_IN)}
          </h1>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
