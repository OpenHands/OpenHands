import { RuntimeWaitingState } from "#/components/features/conversation-panel/runtime-waiting-state";

interface WaitingForRuntimeMessageProps {
  className?: string;
  testId?: string;
}

export function WaitingForRuntimeMessage({
  className,
  testId,
}: WaitingForRuntimeMessageProps) {
  return <RuntimeWaitingState testId={testId} className={className} />;
}
