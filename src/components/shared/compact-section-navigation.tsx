import { useId, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { useNavigation } from "#/context/navigation-context";
import {
  SETTINGS_COMPACT_MAX_WIDTH,
  SIDEBAR_RAIL_COLLAPSE_MAX_WIDTH,
  useBreakpoint,
} from "#/hooks/use-breakpoint";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";
import { cn } from "#/utils/utils";
import { dropdownTriggerShellClassName } from "#/utils/dropdown-classes";

interface CompactSectionNavigationProps {
  label: string;
  currentLabel?: string;
  testId: string;
  children: React.ReactNode;
}

function SectionDisclosure({
  label,
  currentLabel,
  testId,
  children,
}: CompactSectionNavigationProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const containerRef = useClickOutsideElement<HTMLDivElement>(() =>
    setIsOpen(false),
  );
  useCloseOnEscape(isOpen, () => setIsOpen(false), triggerRef);

  return (
    <div ref={containerRef} className="relative" data-testid={testId}>
      <button
        ref={triggerRef}
        type="button"
        aria-label={`${label}: ${currentLabel}`}
        aria-expanded={isOpen}
        aria-controls={menuId}
        onClick={() => setIsOpen(!isOpen)}
        className={cn(dropdownTriggerShellClassName, "min-h-11 text-sm")}
      >
        <span className="shrink-0 text-muted">{label}</span>
        <span className="min-w-0 flex-1 truncate text-left text-contrast">
          {currentLabel}
        </span>
        <ChevronDown className="size-4 shrink-0" aria-hidden />
      </button>
      {isOpen && (
        // Link activation bubbles here for mouse and keyboard alike. Child
        // anchors own navigation; this only dismisses their disclosure.
        // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions -- delegated link activation, not a standalone click target
        <nav
          id={menuId}
          aria-label={label}
          className="absolute inset-x-0 top-full z-50 mt-1 max-h-80 overflow-y-auto rounded-md border border-border-subtle bg-tertiary p-1 shadow-lg"
          onClick={(event) => {
            if (event.target instanceof Element && event.target.closest("a")) {
              setIsOpen(false);
            }
          }}
        >
          {children}
        </nav>
      )}
    </div>
  );
}

/** Keep sibling pages reachable while the secondary sidebar is hidden. */
export function CompactSectionNavigation(props: CompactSectionNavigationProps) {
  const { currentPath } = useNavigation();
  const isCompact = useBreakpoint(SETTINGS_COMPACT_MAX_WIDTH);
  const isPhone = useBreakpoint(SIDEBAR_RAIL_COLLAPSE_MAX_WIDTH);

  if (!isCompact || isPhone || !props.currentLabel) return null;

  // Route changes also dismiss a disclosure opened before browser Back/Forward.
  return <SectionDisclosure key={currentPath} {...props} />;
}
