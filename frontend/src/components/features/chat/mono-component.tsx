import { ReactNode } from "react";

/**
 * Renders monospaced bold text. Used as a ``<Trans>`` component override
 * for the ``<cmd>`` tag in translation strings. Receives children verbatim
 * — i18next no longer double-escapes interpolated values (see
 * ``i18n/index.ts``'s ``escapeValue: false``), so the prior
 * ``decodeHtmlEntities`` step had nothing to undo.
 */
function MonoComponent({ children }: { children?: ReactNode }) {
  return <strong className="font-mono">{children}</strong>;
}

export { MonoComponent };
