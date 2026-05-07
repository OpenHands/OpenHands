import { Spinner } from "./spinner";

interface LoaderProps {
  size?: "xs" | "sm" | "md" | "lg";
  className?: string;
}

export function Loader({ size = "sm", className }: LoaderProps) {
  return (
    <div data-testid="loader" className={className}>
      <Spinner size={size} />
    </div>
  );
}
