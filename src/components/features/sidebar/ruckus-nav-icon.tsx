import type { SVGProps } from "react";

type RuckusNavIconProps = SVGProps<SVGSVGElement> & {
  kind: "conversations" | "automations" | "customize";
};

/** The three main destinations share the same cut-out, two-tone geometry. */
export function RuckusNavIcon({ kind, ...props }: RuckusNavIconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      aria-hidden="true"
      {...props}
    >
      {kind === "conversations" && (
        <>
          <path d="M3 3h18v14H10l-7 4V3Z" fill="currentColor" opacity=".2" />
          <path
            d="M3 3h18v14H10l-7 4V3Zm4 5h10M7 12h7"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="miter"
          />
        </>
      )}
      {kind === "automations" && (
        <>
          <path
            d="m14 2-9 12h6l-1 8 9-12h-6l1-8Z"
            fill="currentColor"
            opacity=".2"
          />
          <path
            d="m14 2-9 12h6l-1 8 9-12h-6l1-8Z"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinejoin="miter"
          />
        </>
      )}
      {kind === "customize" && (
        <>
          <path
            d="M3 3h7v7H3zm11 11h7v7h-7Z"
            fill="currentColor"
            opacity=".3"
          />
          <path
            d="M3 3h7v7H3zm11 11h7v7h-7ZM17.5 2v9M13 6.5h9M2 17.5h9M6.5 13v9"
            stroke="currentColor"
            strokeWidth="2"
          />
        </>
      )}
    </svg>
  );
}
