import React from "react";
import { FaCircleInfo } from "react-icons/fa6";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BrandButton } from "../../settings/brand-button";
import { SettingsDropdownInput } from "../../settings/settings-dropdown-input";
import { BadgeInput } from "#/components/shared/inputs/badge-input";
import { cn } from "#/utils/utils";
import CloseIcon from "#/icons/close.svg?react";
import { useMicroagentPrompt } from "#/hooks/query/use-microagent-prompt";
import { useHandleRuntimeActive } from "#/hooks/use-handle-runtime-active";
import { LoadingMicroagentBody } from "./loading-microagent-body";
import { LoadingMicroagentTextarea } from "./loading-microagent-textarea";
import { useGetMicroagents } from "#/hooks/query/use-get-microagents";
import { Typography } from "#/ui/typography";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";

interface LaunchMicroagentModalProps {
  onClose: () => void;
  onLaunch: (query: string, target: string, triggers: string[]) => void;
  eventId: number;
  isLoading: boolean;
  selectedRepo: string;
}

export function LaunchMicroagentModal({
  onClose,
  onLaunch,
  eventId,
  isLoading,
  selectedRepo,
}: LaunchMicroagentModalProps) {
  const { t } = useTranslation();
  const { runtimeActive } = useHandleRuntimeActive();
  const { data: conversation } = useActiveConversation();
  const { data: prompt, isLoading: promptIsLoading } =
    useMicroagentPrompt(eventId);

  const { data: microagents, isLoading: microagentsIsLoading } =
    useGetMicroagents(`${selectedRepo}/.openhands/microagents`);

  const [triggers, setTriggers] = React.useState<string[]>([]);

  // TODO: Hide LaunchMicroagentModal for V1 conversations
  // This is a temporary measure and may be re-enabled in the future
  const isV1Conversation = conversation?.conversation_version === "V1";

  // Don't render anything for V1 conversations
  if (isV1Conversation) {
    return null;
  }

  const formAction = (formData: FormData) => {
    const query = formData.get("query-input")?.toString();
    const target = formData.get("target-input")?.toString();

    if (query && target) {
      onLaunch(query, target, triggers);
    }
  };

  const onSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    formAction(formData);
  };

  return (
    <ModalBackdrop onClose={onClose}>
      {!runtimeActive && <LoadingMicroagentBody />}
      {runtimeActive && (
        <ModalBody className="items-start w-[728px]">
          <div className="flex items-center justify-between w-full">
            <h2 className="font-bold text-[20px] leading-6 -tracking-[0.01em] flex items-center gap-2">
              {t("MICROAGENT$ADD_TO_MICROAGENT")}
              <a
                href="https://docs.all-hands.dev/usage/prompting/microagents-overview#microagents-overview"
                target="_blank"
                rel="noopener noreferrer"
              >
                <FaCircleInfo className="text-primary" />
              </a>
            </h2>

            <button type="button" onClick={onClose}>
              <CloseIcon />
            </button>
          </div>

          <Typography.Text className="text-sm text-[#A3A3A3] font-normal leading-5">
            {t("MICROAGENT$DEFINITION")}
          </Typography.Text>

          <form
            data-testid="launch-microagent-modal"
            onSubmit={onSubmit}
            className="flex flex-col gap-6 w-full"
          >
            <label
              htmlFor="query-input"
              className="flex flex-col gap-2.5 w-full text-sm"
            >
              {t("MICROAGENT$WHAT_TO_REMEMBER")}
              {promptIsLoading && <LoadingMicroagentTextarea />}
              {!promptIsLoading && (
                <textarea
                  required
                  data-testid="query-input"
                  name="query-input"
                  defaultValue={prompt}
                  placeholder={t("MICROAGENT$DESCRIBE_WHAT_TO_ADD")}
                  rows={6}
                  className={cn(
                    "bg-tertiary border border-[#717888] w-full rounded p-2 placeholder:italic placeholder:text-tertiary-alt resize-none",
                    "disabled:bg-[#2D2F36] disabled:border-[#2D2F36] disabled:cursor-not-allowed",
                  )}
                />
              )}
            </label>

            <SettingsDropdownInput
              testId="target-input"
              name="target-input"
              label={t("MICROAGENT$WHERE_TO_PUT")}
              placeholder={t("MICROAGENT$SELECT_FILE_OR_CUSTOM")}
              required
              allowsCustomValue
              isLoading={microagentsIsLoading}
              items={
                microagents?.map((item) => ({
                  key: item,
                  label: item,
                })) || []
              }
            />

            <label
              htmlFor="trigger-input"
              className="flex flex-col gap-2.5 w-full text-sm"
            >
              <div className="flex items-center gap-2">
                {t("MICROAGENT$ADD_TRIGGERS")}
                <a
                  href="https://docs.all-hands.dev/usage/prompting/microagents-keyword"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <FaCircleInfo className="text-primary" />
                </a>
              </div>
              <BadgeInput
                name="trigger-input"
                value={triggers}
                placeholder={t("MICROAGENT$TYPE_TRIGGER_SPACE")}
                onChange={setTriggers}
              />
            </label>

            <div className="flex items-center justify-end gap-2">
              <BrandButton type="button" variant="secondary" onClick={onClose}>
                {t("MICROAGENT$CANCEL")}
              </BrandButton>
              <BrandButton
                type="submit"
                variant="primary"
                isDisabled={
                  isLoading || promptIsLoading || microagentsIsLoading
                }
              >
                {t("MICROAGENT$LAUNCH")}
              </BrandButton>
            </div>
          </form>
        </ModalBody>
      )}
    </ModalBackdrop>
  );
}
