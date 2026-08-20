import { BaseModal } from "./base-modal";

interface DangerModalProps {
  testId?: string;

  title: string;
  description: string;

  buttons: {
    danger: { text: string; onClick: () => void };
    cancel: { text: string; onClick: () => void };
  };
}

export function DangerModal({
  testId,
  title,
  description,
  buttons,
}: DangerModalProps) {
  return (
    <BaseModal
      testId={testId}
      title={title}
      description={description}
      buttons={[
        {
          text: buttons.danger.text,
          onClick: buttons.danger.onClick,
          className: "bg-danger",
        },
        {
          text: buttons.cancel.text,
          onClick: buttons.cancel.onClick,
          className: "bg-[var(--oh-interactive-selected)]",
        },
      ]}
    />
  );
}
