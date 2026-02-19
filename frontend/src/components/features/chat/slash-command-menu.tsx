import React from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { SlashCommand } from "#/config/slash-commands";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

interface SlashCommandMenuProps {
  isOpen: boolean;
  commands: SlashCommand[];
  selectedIndex: number;
  onSelectCommand: (command: SlashCommand) => void;
}

export function SlashCommandMenu({
  isOpen,
  commands,
  selectedIndex,
  onSelectCommand,
}: SlashCommandMenuProps) {
  const { t } = useTranslation();
  const isMac = /Mac|iPod|iPhone|iPad/.test(navigator.userAgent);

  if (!isOpen || commands.length === 0) {
    return null;
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 10 }}
        transition={{ duration: 0.15 }}
        className="absolute bottom-full left-0 right-0 mb-2 z-50"
      >
        <div className="bg-[#333333] border border-[#525252] rounded-lg shadow-lg overflow-hidden">
          <div className="px-3 py-2 border-b border-[#525252]">
            <Typography.Text className="text-xs text-[#9A9A9A] font-medium">
              {t(I18nKey.SLASH_COMMANDS$TITLE)}
            </Typography.Text>
          </div>
          <ul className="py-1 max-h-64 overflow-y-auto custom-scrollbar-always">
            {commands.map((command, index) => (
              <li key={command.name}>
                <button
                  type="button"
                  onClick={() => onSelectCommand(command)}
                  className={cn(
                    "w-full px-3 py-2 flex items-center justify-between gap-3 text-left transition-colors",
                    "hover:bg-[#454545]",
                    index === selectedIndex && "bg-[#454545]",
                  )}
                >
                  <div className="flex items-center gap-3">
                    <Typography.Text className="text-[#CDA1FA] font-mono text-sm font-medium">
                      {command.name}
                    </Typography.Text>
                    <Typography.Text className="text-[#9A9A9A] text-sm">
                      {t(command.description)}
                    </Typography.Text>
                  </div>
                  {command.shortcut && (
                    <Typography.Text className="text-[#6B6B6B] text-xs font-mono bg-[#252525] px-1.5 py-0.5 rounded">
                      {isMac ? command.shortcut.mac : command.shortcut.default}
                    </Typography.Text>
                  )}
                </button>
              </li>
            ))}
          </ul>
          <div className="px-3 py-2 border-t border-[#525252] flex gap-4 text-xs text-[#6B6B6B]">
            <Typography.Text className="text-xs text-[#6B6B6B]">
              <kbd className="bg-[#252525] px-1 rounded">↑↓</kbd>{" "}
              {t(I18nKey.SLASH_COMMANDS$NAVIGATE)}
            </Typography.Text>
            <Typography.Text className="text-xs text-[#6B6B6B]">
              <kbd className="bg-[#252525] px-1 rounded">
                {t(I18nKey.SLASH_COMMANDS$KEY_ENTER)}
              </kbd>{" "}
              {t(I18nKey.SLASH_COMMANDS$SELECT)}
            </Typography.Text>
            <Typography.Text className="text-xs text-[#6B6B6B]">
              <kbd className="bg-[#252525] px-1 rounded">
                {t(I18nKey.SLASH_COMMANDS$KEY_ESCAPE)}
              </kbd>{" "}
              {t(I18nKey.SLASH_COMMANDS$CLOSE)}
            </Typography.Text>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
