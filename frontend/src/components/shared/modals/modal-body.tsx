import React from "react";
import { cn } from "#/utils/utils";

type ModalWidth = "small" | "medium";

interface ModalBodyProps {
  testID?: string;
  children: React.ReactNode;
  className?: React.HTMLProps<HTMLDivElement>["className"];
  width?: ModalWidth;
}

export function ModalBody({
  testID,
  children,
  className,
  width = "small",
}: ModalBodyProps) {
  return (
    <div
      data-testid={testID}
      className={cn(
        "modal-body bg-base-secondary flex flex-col gap-6 items-center p-6 rounded-xl",
        width === "small" && "w-[384px]",
        width === "medium" && "w-[700px]",
        className,
      )}
    >
      {children}
    </div>
  );
}
