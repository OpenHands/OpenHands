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
      // The type is already passed as a prop to the button component
      // eslint-disable-next-line react/button-has-type
      type={type}
      onClick={onClick}
      className={cn(
        "w-fit px-4 py-2.5 text-sm rounded-lg font-medium disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer transition-all duration-200",
        variant === "primary" &&
          "bg-gradient-to-r from-[#6366F1] to-[#4F46E5] text-white hover:from-[#818CF8] hover:to-[#6366F1] shadow-[0_1px_3px_rgba(99,102,241,0.3)] hover:shadow-[0_4px_16px_rgba(99,102,241,0.35)] active:scale-[0.98]",
        variant === "secondary" &&
          "border border-[#3F3F46] text-[#E4E4E7] bg-transparent hover:bg-[#18181B] hover:border-[#6366F1]/40",
        variant === "danger" &&
          "bg-[#E11D48] text-white hover:bg-[#BE123C] shadow-sm",
        variant === "ghost-danger" &&
          "bg-transparent text-[#F43F5E] underline hover:text-[#FB7185] hover:no-underline font-medium",
        startContent && "flex items-center justify-center gap-2",
        className,
      )}
    >
      {startContent}
      {children}
    </button>
  );
}
