import { useEffect, useRef, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { LayoutGroup, motion, useReducedMotion } from "framer-motion";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

export type HomeLauncherMode = "code" | "automate";

interface HomeLauncherModeToggleProps {
  mode: HomeLauncherMode;
  onChange: (mode: HomeLauncherMode) => void;
}

const PILL_TRANSITION = {
  type: "spring",
  stiffness: 420,
  damping: 34,
  mass: 0.55,
} as const;

function CodeBracketsIcon({ active }: { active: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="size-4"
      aria-hidden
    >
      <motion.g
        animate={active ? { x: [0, -1.5, 0] } : { x: 0 }}
        transition={
          active ? { duration: 0.38, ease: "easeInOut" } : { duration: 0.15 }
        }
      >
        <path d="m8 6-6 6 6 6" />
      </motion.g>
      <motion.g
        animate={active ? { x: [0, 1.5, 0] } : { x: 0 }}
        transition={
          active ? { duration: 0.38, ease: "easeInOut" } : { duration: 0.15 }
        }
      >
        <path d="m16 18 6-6-6-6" />
      </motion.g>
    </svg>
  );
}

function AutomateClockIcon({ active }: { active: boolean }) {
  const handRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const node = handRef.current;
    if (!node || !active || import.meta.env.MODE === "test") {
      return undefined;
    }

    const animation = node.animate(
      [{ transform: "rotate(0deg)" }, { transform: "rotate(360deg)" }],
      {
        duration: 700,
        easing: "cubic-bezier(0.2, 0.7, 0.2, 1)",
        fill: "forwards",
      },
    );

    return () => {
      animation.cancel();
    };
  }, [active]);

  return (
    <span className="relative inline-flex size-4" aria-hidden>
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="absolute inset-0 size-4"
      >
        <circle cx="12" cy="12" r="9" />
        <path d="M12 12V8" />
      </svg>
      <span ref={handRef} className="absolute inset-0 size-4 origin-center">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="size-4"
        >
          <path d="M12 12L16.5 15" />
        </svg>
      </span>
    </span>
  );
}

export function HomeLauncherModeToggle({
  mode,
  onChange,
}: HomeLauncherModeToggleProps) {
  const { t } = useTranslation("openhands");
  const reduceMotion = useReducedMotion();
  const disableAnimation = reduceMotion || import.meta.env.MODE === "test";

  const options: Array<{
    value: HomeLauncherMode;
    label: string;
    icon: (active: boolean) => ReactNode;
  }> = [
    {
      value: "code",
      label: t(I18nKey.COMMON$CODE),
      icon: (active) => (
        <CodeBracketsIcon active={active && !disableAnimation} />
      ),
    },
    {
      value: "automate",
      label: t(I18nKey.AUTOMATE$SECTION_TITLE),
      icon: (active) => <AutomateClockIcon active={active} />,
    },
  ];

  return (
    <div className="flex w-full justify-center">
      <LayoutGroup id="home-launch-mode">
        <div
          role="radiogroup"
          aria-label={t(I18nKey.HOME$MODE_TOGGLE_LABEL)}
          data-testid="home-launcher-mode-toggle"
          className="inline-flex items-center rounded-full bg-[var(--oh-surface-raised)] p-1 text-sm font-medium"
        >
          {options.map((option) => {
            const isActive = option.value === mode;
            return (
              <button
                key={option.value}
                type="button"
                role="radio"
                aria-checked={isActive}
                data-testid={`home-launcher-mode-${option.value}`}
                onClick={() => onChange(option.value)}
                className={cn(
                  "relative inline-flex items-center gap-2 rounded-full px-5 py-2",
                  "cursor-pointer transition-colors",
                  isActive
                    ? "text-white"
                    : "text-[var(--oh-muted)] hover:text-white",
                  disableAnimation &&
                    isActive &&
                    "bg-[var(--oh-interactive-hover)]",
                )}
              >
                {!disableAnimation && isActive ? (
                  <motion.span
                    layoutId="home-launch-mode-pill"
                    className="absolute inset-0 rounded-full bg-[var(--oh-interactive-hover)]"
                    transition={PILL_TRANSITION}
                  />
                ) : null}
                <span className="relative z-10 inline-flex size-4 shrink-0 items-center justify-center">
                  {option.icon(isActive)}
                </span>
                <span className="relative z-10">{option.label}</span>
              </button>
            );
          })}
        </div>
      </LayoutGroup>
    </div>
  );
}
