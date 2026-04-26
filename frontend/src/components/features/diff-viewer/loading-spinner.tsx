import { Spinner } from "#/components/shared/spinner";

export interface LoadingSpinnerProps {
  className?: string;
}

export function LoadingSpinner({ className }: LoadingSpinnerProps) {
  return <Spinner size="md" spinnerClassName={className} />;
}
