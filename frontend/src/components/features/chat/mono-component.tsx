import { ReactNode } from "react";
import EventLogger from "#/utils/event-logger";

const MAX_DISPLAY_LENGTH = 80;

const decodeHtmlEntities = (text: string): string => {
  const textarea = document.createElement("textarea");
  textarea.innerHTML = text;
  return textarea.value;
};

function MonoComponent(props: { children?: ReactNode }) {
  const { children } = props;

  const processString = (str: string): ReactNode => {
    try {
      const decoded = decodeHtmlEntities(str);
      const isTruncated = decoded.length > MAX_DISPLAY_LENGTH;
      const displayText = isTruncated
        ? `${decoded.substring(0, MAX_DISPLAY_LENGTH)}...`
        : decoded;

      if (isTruncated) {
        return (
          <span className="font-mono cursor-help" title={decoded}>
            {displayText}
          </span>
        );
      }

      return <span className="font-mono">{displayText}</span>;
    } catch (e) {
      EventLogger.error(String(e));
      return <span className="font-mono">{str}</span>;
    }
  };

  if (Array.isArray(children)) {
    const processedChildren = children.map((child, index) =>
      typeof child === "string" ? (
        <span key={index}>{processString(child)}</span>
      ) : (
        child
      ),
    );

    return <strong className="font-mono">{processedChildren}</strong>;
  }

  if (typeof children === "string") {
    return <strong>{processString(children)}</strong>;
  }

  return <strong className="font-mono">{children}</strong>;
}

export { MonoComponent };
