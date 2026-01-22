import { Dispatch, SetStateAction, useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { validateEmail } from "#/utils/auth";
import WarningIcon from "#/assets/warning.svg?react";

interface EnterEmailOTPProps {
  setIsLogginInWithEmail: Dispatch<SetStateAction<boolean>>;
  setIsReadyToVerifyEmail: Dispatch<SetStateAction<boolean>>;
}

function EnterEmailOTP({
  setIsLogginInWithEmail,
  setIsReadyToVerifyEmail,
}: EnterEmailOTPProps) {
  const { t } = useTranslation();
  const [invalidEmailError, setInvalidEmailError] = useState("");
  const [email, setEmail] = useState("");

  // TODO:: trigger send user email with OTP on submit
  const handleSubmit = () => {
    const { valid, error } = validateEmail(email);

    if (error) {
      setInvalidEmailError(error);
      return;
    }

    if (valid) {
      setInvalidEmailError("");
      setIsReadyToVerifyEmail(true); // render OTP screen
    }
  };

  const buttonLabelClasses = "text-sm font-medium leading-5 px-1";

  return (
    <>
      <span className="text-[39px] font-medium text-center">
        {t(I18nKey.AUTH$SIGN_IN_WITH_EMAIL)}
      </span>

      <form
        className="flex flex-col gap-4 w-full"
        onSubmit={(e) => {
          e.preventDefault();
          handleSubmit();
        }}
      >
        <div className="flex flex-col gap-[6px] w-full">
          <label className="font-[400] text-[14px]">
            {t(I18nKey.AUTH$YOUR_EMAIL_ADDRESS)}
            <input
              data-test-id="email-address-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="name@email.com"
              className="w-full h-[40px] bg-[#454545] border border-solid rounded-[5px] px-[10px] pl-[15px]"
            />
          </label>
        </div>

        {invalidEmailError && (
          <div className="flex items-center bg-[#4A0709] border border-[#FF0006] w-full h-[40px] rounded-[4px] gap-[14px] px-[14px]">
            <WarningIcon />
            <span>{t(invalidEmailError)}</span>
          </div>
        )}

        <div className="flex gap-[6px] h-[40px] text-[14px] w-full">
          <button
            type="submit"
            className="flex-1 bg-[#FFFFFF] rounded-[4px] text-[#000000]"
          >
            <span className={buttonLabelClasses}>{t(I18nKey.AUTH$VERIFY)}</span>
          </button>

          <button
            type="button"
            onClick={() => setIsLogginInWithEmail(false)}
            className="flex-1 border border-[#FFFFFF] rounded-[4px]"
          >
            <span className={buttonLabelClasses}>{t(I18nKey.AUTH$CANCEL)}</span>
          </button>
        </div>
      </form>
    </>
  );
}

export default EnterEmailOTP;
