import { Dispatch, SetStateAction, useState, useRef } from "react";
import { useTranslation } from "react-i18next";
import WarningIcon from "#/assets/warning.svg?react";

interface VerifyEmailOTPPProps {
  setIsOTPComplete: Dispatch<SetStateAction<boolean>>;
  setIsReadyToVerifyEmail: Dispatch<SetStateAction<boolean>>;
}

function VerifyEmailOTP({
  setIsOTPComplete,
  setIsReadyToVerifyEmail,
}: VerifyEmailOTPPProps) {
  const { t } = useTranslation();
  const [otp, setOtp] = useState<string[]>(Array(6).fill(""));
  const [invalidEmailError, setInvalidEmailError] = useState("");

  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  const resetForm = () => {
    // reset OTP inputs
    setOtp(Array(6).fill(""));

    // focus first input after reset
    requestAnimationFrame(() => {
      inputRefs.current[0]?.focus();
    });
  };

  const verifyOTP = (otpValue: string) => {
    // TODO: validate OTP -> compare input to generated OTP value
    const isOTPValid = otpValue;
    if (isOTPValid) {
      setIsOTPComplete(true); // render login org selecter
    } else {
      setInvalidEmailError("Invalid code, please try again");
      resetForm();
    }
  };

  const handleChange = (index: number, value: string) => {
    if (!/^\d?$/.test(value)) return;

    const next = [...otp];
    next[index] = value;
    setOtp(next);

    // auto-focus next input
    if (value && index < inputRefs.current.length - 1) {
      inputRefs.current[index + 1]?.focus();
    }
  };

  const handleSubmit = () => {
    const otpValue = otp.join("");
    verifyOTP(otpValue);
  };

  const buttonLabelClasses = "text-sm font-medium leading-5 px-1";

  return (
    <>
      <span className="text-[39px] font-medium text-center">
        {t("Enter Code")}
      </span>

      <span className="text-[14px] font-[400] text-[#A3A3A3] w-[355px] text-center">
        {t(
          "An OTP code has been sent to your email address. Enter that code here. Didn’t receive it? Resend Code (link)",
        )}
      </span>

      <form
        className="flex flex-col gap-[31px] w-full items-center"
        onSubmit={(e) => {
          e.preventDefault();
          handleSubmit();
        }}
      >
        {/* OTP inputs */}
        <div className="flex gap-[8px] w-[210px] mx-auto">
          {otp.map((value, index) => (
            <input
              key={index}
              ref={(el) => {
                inputRefs.current[index] = el;
              }}
              value={value}
              onChange={(e) => handleChange(index, e.target.value)}
              inputMode="numeric"
              maxLength={1}
              className="flex-1 bg-[#454545] border border-[#727987] w-[30px] h-[40px] rounded-[5px] text-center"
            />
          ))}
        </div>

        {invalidEmailError && (
          <div className="flex items-center bg-[#4A0709] border border-[#FF0006] w-full h-[40px] rounded-[4px] gap-[14px] px-[14px]">
            <WarningIcon />
            <span>{t(invalidEmailError)}</span>
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-[6px] h-[40px] text-[14px] w-full">
          <button
            type="submit"
            className="flex-1 bg-[#FFFFFF] rounded-[4px] text-[#000000]"
          >
            <span className={buttonLabelClasses}>{t("Next")}</span>
          </button>

          <button
            type="button"
            onClick={() => setIsReadyToVerifyEmail(false)}
            className="flex-1 border border-[#FFFFFF] rounded-[4px]"
          >
            <span className={buttonLabelClasses}>{t("Back")}</span>
          </button>
        </div>
      </form>
    </>
  );
}

export default VerifyEmailOTP;
