import { cn } from "#/utils/utils";

interface BrandButtonProps {
  testId?: string;
  name?: string;
  variant: "primary" | "secondary" | "danger" | "ghost-danger";
  type: React.ButtonHTMLAttributes<HTMLButtonElement>["type"];
  isDisabled?: boolean;
  className?: string;
  onClick?: () => void;
  startContent?: React.ReactNode;
}

export function BrandButton({
  testId,
  name,
  children,
  variant,
  type,
  isDisabled,
  className,
  onClick,
  startContent,
}: React.PropsWithChildren<BrandButtonProps>) {
  return (
    <button
      name={name}
      data-testid={testId}
      disabled={isDisabled}
      // The type is alreadt passed as a prop to the button component
      // eslint-disable-next-line react/button-has-type
      type={type}
      onClick={onClick}
      className={cn(
        "w-fit p-2 text-sm rounded-sm disabled:opacity-30 disabled:cursor-not-allowed hover:opacity-80 cursor-pointer",
        variant === "primary" && "bg-primary text-base",
        variant === "secondary" && "border border-primary text-primary",
        variant === "danger" && "bg-danger text-content hover:bg-danger-dark",
        variant === "ghost-danger" &&
          "bg-transparent text-danger underline hover:text-danger-dark hover:no-underline font-medium",
        startContent && "flex items-center justify-center gap-2",
        className,
      )}
    >
      {startContent}
      {children}
    </button>
  );
}
