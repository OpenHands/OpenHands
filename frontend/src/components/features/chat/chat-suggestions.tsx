import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";

export function ChatSuggestions() {
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
          <div className="flex flex-col items-center p-4 rounded-xl w-full text-center">
            <span className="text-[32px] font-bold leading-10 text-content pt-4 pb-6">
              {t(I18nKey.LANDING$TITLE)}
              <br />
              {t(I18nKey.LANDING$SUBTITLE_PROMPT)}
            </span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
