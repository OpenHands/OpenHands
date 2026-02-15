import { NavLink } from "react-router";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

export function OpenHandsLogoButton() {
  return (
    <StyledTooltip content="neww.ai">
      <NavLink
        to="/"
        aria-label="neww.ai home"
        className="group flex items-center gap-2 transition-opacity hover:opacity-80"
      >
        <div className="flex items-center justify-center w-[42px] h-[28px] rounded-lg bg-gradient-to-br from-[#6366F1] to-[#4F46E5] shadow-[0_2px_8px_rgba(99,102,241,0.3)] group-hover:shadow-[0_4px_16px_rgba(99,102,241,0.4)] transition-shadow">
          <span
            className="text-white font-bold text-[13px] tracking-tight leading-none"
            style={{ fontFamily: "Inter, -apple-system, sans-serif" }}
          >
            n.
          </span>
        </div>
      </NavLink>
    </StyledTooltip>
  );
}
