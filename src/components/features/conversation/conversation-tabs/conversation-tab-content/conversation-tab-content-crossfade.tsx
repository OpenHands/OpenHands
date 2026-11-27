import {
  ReactNode,
  Suspense,
  useCallback,
  useLayoutEffect,
  useState,
} from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ConversationLoading } from "../../conversation-loading";
import { TabReadyNotifier } from "./tab-ready-notifier";
import { SuspensePendingFallback } from "./suspense-pending-fallback";

const CROSSFADE_DURATION_SECONDS = 0.35;

const crossfadeTransition = {
  duration: CROSSFADE_DURATION_SECONDS,
  ease: "easeInOut" as const,
};

type ConversationTabContentCrossfadeProps = {
  showAgentLoading: boolean;
  tabKey: string;
  children: ReactNode;
};

export function ConversationTabContentCrossfade({
  showAgentLoading,
  tabKey,
  children,
}: ConversationTabContentCrossfadeProps) {
  const reduceMotion = useReducedMotion();
  const [lazyPending, setLazyPending] = useState(false);

  useLayoutEffect(() => {
    setLazyPending(false);
  }, [tabKey]);

  const handleLazyPending = useCallback(() => {
    setLazyPending(true);
  }, []);

  const handleLazyReady = useCallback(() => {
    setLazyPending(false);
  }, []);

  const showLoadingOverlay = showAgentLoading || lazyPending;

  if (reduceMotion) {
    return (
      <div className="relative h-full w-full overflow-hidden">
        {showLoadingOverlay ? (
          <ConversationLoading />
        ) : (
          <Suspense
            fallback={<SuspensePendingFallback onPending={handleLazyPending} />}
          >
            <TabReadyNotifier onReady={handleLazyReady}>
              {children}
            </TabReadyNotifier>
          </Suspense>
        )}
      </div>
    );
  }

  return (
    <div className="relative h-full w-full overflow-hidden">
      <motion.div
        className="absolute inset-0 h-full w-full"
        initial={false}
        animate={{ opacity: showLoadingOverlay ? 0 : 1 }}
        transition={crossfadeTransition}
        aria-hidden={showLoadingOverlay}
      >
        <Suspense
          fallback={<SuspensePendingFallback onPending={handleLazyPending} />}
        >
          <TabReadyNotifier onReady={handleLazyReady}>
            {children}
          </TabReadyNotifier>
        </Suspense>
      </motion.div>

      <AnimatePresence>
        {showLoadingOverlay ? (
          <motion.div
            key="conversation-tab-loading-overlay"
            className="absolute inset-0 z-10 h-full w-full"
            initial={{ opacity: 1 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={crossfadeTransition}
          >
            <ConversationLoading />
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
