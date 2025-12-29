import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { Suggestions } from "#/components/features/suggestions/suggestions";
import { I18nKey } from "#/i18n/declaration";
import BuildIt from "#/icons/build-it.svg?react";
import { SUGGESTIONS } from "#/utils/suggestions";
import { useConversationStore } from "#/stores/conversation-store";

interface ChatSuggestionsProps {
  onSuggestionsClick: (value: string) => void;
}

export function ChatSuggestions({ onSuggestionsClick }: ChatSuggestionsProps) {
  const { t } = useTranslation();
  const { shouldHideSuggestions } = useConversationStore();

  return (
    <AnimatePresence>
      {!shouldHideSuggestions && (
        <motion.div
          data-testid="chat-suggestions"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
          className="absolute top-0 left-0 right-0 bottom-[151px] flex flex-col items-center justify-center pointer-events-auto"
        >
          <div className="flex flex-col items-center p-4 rounded-xl w-full">
            <BuildIt width={86} height={103} />
            <span className="text-[32px] font-bold leading-5 text-white pt-4 pb-6">
              {t(I18nKey.LANDING$TITLE)}
            </span>
          </div>
          <Suggestions
            suggestions={Object.entries(SUGGESTIONS.repo)
              .slice(0, 4)
              .map(([label, value]) => ({
                label,
                value,
              }))}
            onSuggestionClick={onSuggestionsClick}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
