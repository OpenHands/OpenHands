import React from "react";

// Custom heading components for plan views with reduced font sizes and tighter spacing
export const planHeadings = {
  h1: ({ children }: { children?: React.ReactNode }) => (
    <h1 className="text-lg text-white font-bold leading-6 mb-1.5 mt-3 first:mt-0">
      {children}
    </h1>
  ),
  h2: ({ children }: { children?: React.ReactNode }) => (
    <h2 className="text-base font-semibold leading-5 text-white mb-1 mt-2.5 first:mt-0">
      {children}
    </h2>
  ),
  h3: ({ children }: { children?: React.ReactNode }) => (
    <h3 className="text-sm font-semibold text-white mb-1 mt-2 first:mt-0">
      {children}
    </h3>
  ),
  h4: ({ children }: { children?: React.ReactNode }) => (
    <h4 className="text-sm font-semibold text-white mb-1 mt-2 first:mt-0">
      {children}
    </h4>
  ),
  h5: ({ children }: { children?: React.ReactNode }) => (
    <h5 className="text-xs font-semibold text-white mb-0.5 mt-1.5 first:mt-0">
      {children}
    </h5>
  ),
  h6: ({ children }: { children?: React.ReactNode }) => (
    <h6 className="text-xs font-medium text-gray-300 mb-0.5 mt-1.5 first:mt-0">
      {children}
    </h6>
  ),
};
