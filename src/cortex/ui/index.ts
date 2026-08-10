/**
 * Cortex UI Design Tokens
 * Rebrands the user interface with premium colors, gradients, and custom branding constants.
 */

export const CORTEX_BRAND = {
  NAME: "CORTEX",
  SLOGAN: "The Agentic Orchestration Engine",
  VERSION: "1.0.0",
};

export const CORTEX_THEME = {
  colors: {
    primary: "rgb(99, 102, 241)", // Indigo Accent
    secondary: "rgb(16, 185, 129)", // Emerald Success Accent
    backgroundDark: "#0B0F19", // Deep Space base background
    surfaceDark: "#151B26", // Deep Space raised surface
    border: "rgba(255, 255, 255, 0.08)",
  },
  animations: {
    fadeIn: "transition-opacity duration-200 ease-in-out",
    slideIn: "transition-transform duration-200 ease-out",
  },
};
