import { Dispatch, SetStateAction, useState } from "react";
import { TermsAndPrivacyNotice } from "#/components/shared/terms-and-privacy-notice";
import EnterEmailOTP from "./enter-email-otp";
import VerifyEmailOTP from "./verify-email-otp";
import LoginOrgSelector from "./login-org-selector";
import TOSReview from "../tos-review";
import FullScreenModal from "#/components/shared/modals/full-screen-modal";
import LoadingBar from "../../loading-bar";

interface LoginWithEmailProps {
  setIsLogginInWithEmail: Dispatch<SetStateAction<boolean>>;
}

function LoginWithEmail({ setIsLogginInWithEmail }: LoginWithEmailProps) {
  const [isReadyToVerifyEmail, setIsReadyToVerifyEmail] = useState(false);
  const [isOrgSelected, setIsOrgSelected] = useState(false);
  const [isOTPComplete, setIsOTPComplete] = useState(false);
  const [isTOSReviewComplete, setIsTOSReviewComplete] = useState(false);

  // TODO:: complete login logic & update when/where LoadingBar is rendered
  // const handleLogin = () => {
  // trigger login methods when all steps are complete
  // returns loading state
  // redirect to home screen
  // }

  const isUserLoginReady =
    isOTPComplete && isOrgSelected && isTOSReviewComplete;

  return (
    <div>
      {isUserLoginReady ? (
        <FullScreenModal header={null}>
          <LoadingBar />
        </FullScreenModal>
      ) : (
        <FullScreenModal
          footer={
            !isOTPComplete ? (
              <TermsAndPrivacyNotice className="max-w-[320px] text-[#A3A3A3]" />
            ) : undefined
          }
        >
          {/* Step 1: Enter Email */}
          {!isOTPComplete && !isReadyToVerifyEmail && (
            <EnterEmailOTP
              setIsLogginInWithEmail={setIsLogginInWithEmail}
              setIsReadyToVerifyEmail={setIsReadyToVerifyEmail}
            />
          )}

          {/* Step 2: Verify Email via OTP */}
          {!isOTPComplete && isReadyToVerifyEmail && (
            <VerifyEmailOTP
              setIsReadyToVerifyEmail={setIsReadyToVerifyEmail}
              setIsOTPComplete={setIsOTPComplete}
            />
          )}

          {/* Step 3: Org Selection */}
          {isOTPComplete && !isOrgSelected && (
            <LoginOrgSelector setIsOrgSelected={setIsOrgSelected} />
          )}

          {/* Step 4: TOS Review */}
          {isOTPComplete && isOrgSelected && !isTOSReviewComplete && (
            <TOSReview
              setIsOrgSelected={setIsOrgSelected}
              setIsTOSReviewComplete={setIsTOSReviewComplete}
            />
          )}
        </FullScreenModal>
      )}
    </div>
  );
}

export default LoginWithEmail;
