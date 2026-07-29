import { ReactNode, useLayoutEffect } from "react";

export function TabReadyNotifier({
  children,
  onReady,
}: {
  children: ReactNode;
  onReady: () => void;
}) {
  useLayoutEffect(() => {
    onReady();
  }, [onReady]);

  return children;
}
