import { LoaderCircle } from "lucide-react";
import { cn } from "#/utils/utils";

export type SpinnerSize = "sm" | "md" | "lg" | "xl";

interface SpinnerProps {
  /**
   * Size of the spinner
   * - sm: 16px (w-4 h-4)
   * - md: 24px (w-6 h-6) - default
   * - lg: 32px (w-8 h-8)
   * - xl: 48px (w-12 h-12)
   */
  size?: SpinnerSize;
  /**
   * Optional label text displayed below the spinner
   */
  label?: string;
  /**
   * Additional CSS classes for the spinner container
   */
  className?: string;
  /**
   * Color of the spinner. Defaults to current text color
   */
  color?: string;
  /**
   * Test ID for testing purposes
   */
  "data-testid"?: string;
}

/**
 * Unified Spinner component to replace multiple loading spinner implementations.
 * 
 * This component consolidates the various spinner implementations found across
 * the codebase:
 * - AgentLoading (uses Lucide LoaderCircle)
 * - ConversationLoading (uses Lucide LoaderCircle with text)
 * - LoadingMicroagentBody (uses HeroUI Spinner)
 * - BranchLoadingState (uses HeroUI Spinner)
 * - RepositoryLoadingState (uses HeroUI Spinner)
 * - SkillsLoadingState (uses CSS border animation)
 * - Home LoadingSpinner (uses CSS border animation)
 * 
 * Migration plan:
 * - Replace AgentLoading: <Spinner size="sm" color="white" />
 * - Replace ConversationLoading: <Spinner size="md" label="Loading..." />
 * - Replace LoadingMicroagentBody: <Spinner size="lg" />
 * - Replace BranchLoadingState: <Spinner size="md" />
 * - Replace RepositoryLoadingState: <Spinner size="lg" />
 * - Replace SkillsLoadingState: <Spinner size="md" />
 * - Replace Home LoadingSpinner: <Spinner size="lg" />
 * 
 * @example
 * ```tsx
 * // Basic usage
 * <Spinner />
 * 
 * // With label
 * <Spinner label="Loading conversation..." />
 * 
 * // Custom size
 * <Spinner size="xl" />
 * 
 * // Custom color
 * <Spinner color="#6366f1" />
 * ```
 */
export function Spinner({
  size = "md",
  label,
  className,
  color,
  "data-testid": testId = "spinner",
}: SpinnerProps) {
  const sizeClasses: Record<SpinnerSize, string> = {
    sm: "w-4 h-4",
    md: "w-6 h-6",
    lg: "w-8 h-8",
    xl: "w-12 h-12",
  };

  const labelSizeClasses: Record<SpinnerSize, string> = {
    sm: "text-xs",
    md: "text-sm",
    lg: "text-base",
    xl: "text-lg",
  };

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2",
        className
      )}
      data-testid={testId}
    >
      <LoaderCircle
        className={cn("animate-spin", sizeClasses[size])}
        style={{ color }}
      />
      {label && (
        <span className={cn("text-neutral-600", labelSizeClasses[size])}>
          {label}
        </span>
      )}
    </div>
  );
}
