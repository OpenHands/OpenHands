import React from "react";
import { CopyToClipboardButton } from "./copy-to-clipboard-button";

export function CopyableContentWrapper({
  text,
  children,
}: {
  text: string;
  children: React.ReactNode;
}) {
  const [isHovering, setIsHovering] = React.useState(false);
  const [isCopied, setIsCopied] = React.useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setIsCopied(true);
  };

  React.useEffect(() => {
    let timeout: NodeJS.Timeout;
    if (isCopied) {
      timeout = setTimeout(() => setIsCopied(false), 2000);
    }
    return () => clearTimeout(timeout);
  }, [isCopied]);

  return (
    <div
      className="relative"
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      <div className="absolute top-2 right-2 z-10">
        <CopyToClipboardButton
          isHidden={!isHovering}
          isDisabled={isCopied}
          onClick={handleCopy}
          mode={isCopied ? "copied" : "copy"}
        />
      </div>
      {children}
    </div>
  );
}
