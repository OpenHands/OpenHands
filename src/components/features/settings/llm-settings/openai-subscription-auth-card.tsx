import React from "react";
import { useTranslation } from "react-i18next";
import { ExternalLink } from "lucide-react";
import type { LLMSubscriptionDeviceChallenge } from "#/api/llm-subscription-service";
import { BrandButton } from "#/components/features/settings/brand-button";
import { CopyToClipboardButton } from "#/components/shared/buttons/copy-to-clipboard-button";
import {
  useLogoutOpenAISubscription,
  usePollOpenAISubscriptionLogin,
  useStartOpenAISubscriptionLogin,
} from "#/hooks/mutation/use-llm-subscription-auth";
import { useOpenAISubscriptionStatus } from "#/hooks/query/use-llm-subscription-status";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

const DEFAULT_DEVICE_POLL_INTERVAL_SECONDS = 5;
const MIN_DEVICE_POLL_INTERVAL_SECONDS = 1;

interface OpenAISubscriptionAuthCardProps {
  isDisabled?: boolean;
}

function openVerificationUrl(challenge: LLMSubscriptionDeviceChallenge) {
  const url = challenge.verificationUriComplete ?? challenge.verificationUri;
  window.open(url, "_blank", "noopener,noreferrer");
}

export function OpenAISubscriptionAuthCard({
  isDisabled = false,
}: OpenAISubscriptionAuthCardProps) {
  const { t } = useTranslation("openhands");
  const status = useOpenAISubscriptionStatus();
  const startLogin = useStartOpenAISubscriptionLogin();
  const pollLogin = usePollOpenAISubscriptionLogin();
  const logout = useLogoutOpenAISubscription();
  const [challenge, setChallenge] =
    React.useState<LLMSubscriptionDeviceChallenge | null>(null);
  const [copied, setCopied] = React.useState(false);
  const [isPendingLogin, setIsPendingLogin] = React.useState(false);
  const pollTimeoutRef = React.useRef<number | null>(null);

  const clearPollTimeout = React.useCallback(() => {
    if (pollTimeoutRef.current !== null) {
      window.clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
  }, []);

  const handleCopyCode = () => {
    if (!challenge) return;
    navigator.clipboard.writeText(challenge.userCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isBusy =
    startLogin.isPending || pollLogin.isPending || logout.isPending;
  const connected = Boolean(status.data?.connected);

  const pollDeviceLogin = React.useCallback(
    async (deviceCode: string) => {
      try {
        const nextStatus = await pollLogin.mutateAsync(deviceCode);
        if (nextStatus.connected) {
          setChallenge(null);
          setIsPendingLogin(false);
          displaySuccessToast(t(I18nKey.SETTINGS$SUBSCRIPTION_CONNECTED_TOAST));
          return true;
        }
        setIsPendingLogin(true);
        return false;
      } catch {
        displayErrorToast(t(I18nKey.SETTINGS$SUBSCRIPTION_CONNECT_ERROR));
        return true;
      }
    },
    [pollLogin, t],
  );

  React.useEffect(() => {
    if (!challenge || connected || isDisabled) {
      return undefined;
    }

    let cancelled = false;
    const intervalSeconds = Math.max(
      challenge.intervalSeconds ?? DEFAULT_DEVICE_POLL_INTERVAL_SECONDS,
      MIN_DEVICE_POLL_INTERVAL_SECONDS,
    );

    const schedulePoll = () => {
      clearPollTimeout();
      pollTimeoutRef.current = window.setTimeout(async () => {
        if (cancelled) {
          return;
        }

        const shouldStop = await pollDeviceLogin(challenge.deviceCode);
        if (!cancelled && !shouldStop) {
          schedulePoll();
        }
      }, intervalSeconds * 1000);
    };

    schedulePoll();

    return () => {
      cancelled = true;
      clearPollTimeout();
    };
  }, [challenge, clearPollTimeout, connected, isDisabled, pollDeviceLogin]);

  const handleStartLogin = async () => {
    try {
      const nextChallenge = await startLogin.mutateAsync();
      setChallenge(nextChallenge);
      setIsPendingLogin(true);
      openVerificationUrl(nextChallenge);
    } catch {
      displayErrorToast(t(I18nKey.SETTINGS$SUBSCRIPTION_CONNECT_ERROR));
    }
  };

  const handlePollLogin = async () => {
    if (!challenge) return;
    clearPollTimeout();
    void pollDeviceLogin(challenge.deviceCode);
  };

  const handleLogout = async () => {
    try {
      await logout.mutateAsync();
      clearPollTimeout();
      setChallenge(null);
      setIsPendingLogin(false);
      displaySuccessToast(t(I18nKey.SETTINGS$SUBSCRIPTION_DISCONNECTED_TOAST));
    } catch {
      displayErrorToast(t(I18nKey.ERROR$GENERIC));
    }
  };

  const handleCancelLogin = () => {
    clearPollTimeout();
    setChallenge(null);
    setIsPendingLogin(false);
  };

  return (
    <section
      data-testid="openai-subscription-auth-card"
      className="flex flex-col gap-4 rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] p-4"
    >
      <div className="flex flex-col gap-2">
        <Typography.H3>
          {t(I18nKey.SETTINGS$SUBSCRIPTION_CARD_TITLE)}
        </Typography.H3>
        <Typography.Paragraph className="text-tertiary-alt text-sm leading-5">
          {t(I18nKey.SETTINGS$SUBSCRIPTION_CARD_DESCRIPTION)}
        </Typography.Paragraph>
      </div>

      <div
        className="flex flex-col gap-1 text-sm"
        data-testid="subscription-status"
      >
        {status.isLoading ? (
          <span className="text-tertiary-light">
            {t(I18nKey.SETTINGS$SUBSCRIPTION_STATUS_CHECKING)}
          </span>
        ) : status.isError ? (
          <span className="text-danger">
            {t(I18nKey.SETTINGS$SUBSCRIPTION_STATUS_UNAVAILABLE)}
          </span>
        ) : connected ? (
          <>
            <span className="text-success">
              {t(I18nKey.SETTINGS$SUBSCRIPTION_STATUS_CONNECTED)}
            </span>
            {status.data?.accountEmail ? (
              <span className="text-tertiary-light">
                {t(I18nKey.SETTINGS$SUBSCRIPTION_ACCOUNT, {
                  account: status.data.accountEmail,
                })}
              </span>
            ) : null}
          </>
        ) : (
          <span className="text-warning">
            {t(I18nKey.SETTINGS$SUBSCRIPTION_STATUS_DISCONNECTED)}
          </span>
        )}
      </div>

      {challenge ? (
        <div
          data-testid="subscription-device-challenge"
          className="flex flex-col gap-2 rounded-lg border border-[var(--oh-border-subtle)] p-3 text-sm"
        >
          <span>{t(I18nKey.SETTINGS$SUBSCRIPTION_DEVICE_INSTRUCTIONS)}</span>
          <div className="flex items-center gap-2 rounded-lg bg-[var(--oh-surface-deep)] px-3 py-2 font-mono text-base font-semibold text-white">
            <span
              data-testid="subscription-user-code"
              className="flex-1 select-all tracking-[0.08em]"
            >
              {challenge.userCode}
            </span>
            <CopyToClipboardButton
              isHidden={false}
              isDisabled={false}
              onClick={handleCopyCode}
              mode={copied ? "copied" : "copy"}
            />
          </div>
          <a
            href={
              challenge.verificationUriComplete ?? challenge.verificationUri
            }
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-sm text-[var(--oh-accent)] underline"
          >
            {t(I18nKey.SETTINGS$SUBSCRIPTION_OPEN_LOGIN)}
            <ExternalLink size={14} aria-hidden />
          </a>
          {isPendingLogin ? (
            <span className="text-warning">
              {t(I18nKey.SETTINGS$SUBSCRIPTION_PENDING_TOAST)}
            </span>
          ) : null}
        </div>
      ) : null}

      <div className="flex flex-wrap gap-2">
        {connected ? (
          <BrandButton
            testId="subscription-disconnect"
            type="button"
            variant="tertiary"
            isDisabled={isDisabled || isBusy}
            onClick={handleLogout}
          >
            {t(I18nKey.SETTINGS$SUBSCRIPTION_DISCONNECT)}
          </BrandButton>
        ) : (
          <BrandButton
            testId="subscription-connect"
            type="button"
            variant="primary"
            isDisabled={isDisabled || isBusy}
            onClick={handleStartLogin}
          >
            {t(I18nKey.SETTINGS$SUBSCRIPTION_CONNECT)}
          </BrandButton>
        )}

        {challenge ? (
          <>
            <BrandButton
              testId="subscription-poll"
              type="button"
              variant="secondary"
              isDisabled={isDisabled || isBusy}
              onClick={handlePollLogin}
            >
              {t(I18nKey.SETTINGS$SUBSCRIPTION_FINISH_SIGN_IN)}
            </BrandButton>
            <BrandButton
              testId="subscription-cancel"
              type="button"
              variant="tertiary"
              isDisabled={isDisabled || isBusy}
              onClick={handleCancelLogin}
            >
              {t(I18nKey.BUTTON$CANCEL)}
            </BrandButton>
          </>
        ) : null}
      </div>
    </section>
  );
}
