import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { METADATA_PREFIXES } from "#/utils/constants";
import { Typography } from "#/ui/typography";
import { MarkdownRenderer } from "../../../features/markdown/markdown-renderer";

interface ParsedSkillContent {
  matchInfo: string | null;
  filePath: string | null;
  body: string;
}

/**
 * Parses skill content into metadata (keyword match info, file path)
 * and the actual skill body.
 */
export function parseSkillContent(content: string): ParsedSkillContent {
  const lines = content.split("\n");
  let matchInfo: string | null = null;
  let filePath: string | null = null;
  let bodyStartIndex = 0;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];

    if (line.startsWith(METADATA_PREFIXES[0])) {
      matchInfo = line.trim();
      bodyStartIndex = i + 1;
    } else if (line.startsWith(METADATA_PREFIXES[2])) {
      filePath = line.replace(METADATA_PREFIXES[2], "").trim();
      bodyStartIndex = i + 1;
    } else if (
      line.startsWith(METADATA_PREFIXES[1]) ||
      line.startsWith(METADATA_PREFIXES[3])
    ) {
      bodyStartIndex = i + 1;
    } else if (line.trim() === "" && i <= bodyStartIndex) {
      bodyStartIndex = i + 1;
    } else {
      break;
    }
  }

  return {
    matchInfo,
    filePath,
    body: lines.slice(bodyStartIndex).join("\n").trim(),
  };
}

/**
 * Wraps `<important>...</important>` tags in the content with bold markers
 * so the markdown renderer displays them as bold text.
 */
export function styleImportantTags(text: string): string {
  return text.replace(
    /<important>([\s\S]*?)<\/important>/gi,
    (_match, inner) => `**${inner.trim()}**`,
  );
}

interface SkillItemExpandedProps {
  content: string;
}

export function SkillItemExpanded({ content }: SkillItemExpandedProps) {
  const { t } = useTranslation();
  const { matchInfo, filePath, body } = parseSkillContent(content);
  const hasMetadata = matchInfo || filePath;

  return (
    <div className="pl-6 pr-2 pt-2 pb-2">
      {hasMetadata && (
        <div className="mb-3 text-sm text-neutral-400 space-y-1">
          {matchInfo && <p>{matchInfo}</p>}
          {filePath && (
            <p>
              <Typography.Text className="text-neutral-500">
                {t(I18nKey.COMMON$PATH)}{" "}
              </Typography.Text>
              <code className="bg-neutral-800 px-1.5 py-0.5 rounded text-neutral-300">
                {filePath}
              </code>
            </p>
          )}
        </div>
      )}

      {hasMetadata && body && <hr className="border-neutral-700 mb-3" />}

      {body && <MarkdownRenderer>{styleImportantTags(body)}</MarkdownRenderer>}
    </div>
  );
}
