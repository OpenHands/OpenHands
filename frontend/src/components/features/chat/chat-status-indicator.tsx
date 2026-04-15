import { cn } from "@heroui/react";
import { motion, AnimatePresence } from "framer-motion";
import DebugStackframeDot from "#/icons/debug-stackframe-dot.svg?react";

interface ChatStatusIndicatorProps {
  status: string;
  statusColor: string;
}

function ChatStatusIndicator({
  status,
  statusColor,
}: ChatStatusIndicatorProps) {
  return (
    <div
      data-testid="chat-status-indicator"
      className={cn(
        "min-h-[31px] w-full max-w-full rounded-[100px] px-4 py-1.5 bg-[#25272D] flex items-center pl-2",
      )}
    >
      <AnimatePresence mode="wait">
        {/* Dot */}
        <motion.span
          key={`dot-${status}`}
          className="flex-shrink-0 animate-[pulse_1.2s_ease-in-out_infinite]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          <DebugStackframeDot className="w-6 h-6" color={statusColor} />
        </motion.span>

        {/* Text */}
        <motion.span
          key={`text-${status}`}
          initial={{ opacity: 0, y: -2 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 2 }}
          transition={{ duration: 0.3 }}
          className="font-normal text-[11px] leading-[16px] normal-case break-words whitespace-normal"
        >
          {status}
        </motion.span>
      </AnimatePresence>
    </div>
  );
}

export default ChatStatusIndicator;
