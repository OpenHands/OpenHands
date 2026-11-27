import { ReactNode } from "react";
import EventLogger from "#/utils/event-logger";

const decodeHtmlEntities = (text: string): string => {
  const textarea = document.createElement("textarea");
  textarea.innerHTML = text;
  return textarea.value;
};

function MonoComponent(props: { children?: ReactNode }) {
  const { children } = props;

  const decodeString = (str: string): string => {
    try {
      return decodeHtmlEntities(str);
    } catch (e) {
      EventLogger.error(String(e));
      return str;
    }
  };

  if (Array.isArray(children)) {
    const processedChildren = children.map((child) =>
      typeof child === "string" ? decodeString(child) : child,
    );

    return <strong className="font-mono">{processedChildren}</strong>;
  }

  if (typeof children === "string") {
    return <strong className="font-mono">{decodeString(children)}</strong>;
  }

  return <strong className="font-mono">{children}</strong>;
}

export { MonoComponent };
