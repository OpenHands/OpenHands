import React from "react";
import { ModalBody } from "../modal-body";
import { ModalButton } from "../../buttons/modal-button";
import { cn } from "#/utils/utils";
import { modalTitleClassName } from "#/utils/modal-classes";

interface ButtonConfig {
  text: string;
  onClick: () => void;
  className: React.HTMLProps<HTMLButtonElement>["className"];
}

interface BaseModalTitleProps {
  title: React.ReactNode;
  id?: string;
  className?: string;
}

export function BaseModalTitle({ title, id, className }: BaseModalTitleProps) {
  return (
    <span id={id} className={cn(modalTitleClassName, className)}>
      {title}
    </span>
  );
}

interface BaseModalDescriptionProps {
  description?: React.ReactNode;
  children?: React.ReactNode;
}

export function BaseModalDescription({
  description,
  children,
}: BaseModalDescriptionProps) {
  return (
    <span className="text-xs text-modal-muted">{children || description}</span>
  );
}

interface BaseModalProps {
  testId?: string;
  title: string;
  description: string;
  buttons: ButtonConfig[];
}

export function BaseModal({
  testId,
  title,
  description,
  buttons,
}: BaseModalProps) {
  return (
    <ModalBody testID={testId}>
      <div className="flex flex-col gap-2 self-start">
        <BaseModalTitle title={title} />
        <BaseModalDescription description={description} />
      </div>

      <div className="flex flex-col gap-2 w-full">
        {buttons.map((button, index) => (
          <ModalButton
            key={index}
            onClick={button.onClick}
            text={button.text}
            className={button.className}
          />
        ))}
      </div>
    </ModalBody>
  );
}
