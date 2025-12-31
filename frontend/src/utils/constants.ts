export const ASSET_FILE_TYPES = [
  ".png",
  ".jpg",
  ".jpeg",
  ".bmp",
  ".gif",
  ".pdf",
  ".mp4",
  ".webm",
  ".ogg",
];

export const JSON_VIEW_THEME = {
  base00: "transparent", // background
  base01: "#2d2d2d", // lighter background
  base02: "#4e4e4e", // selection background
  base03: "#6c6c6c", // comments, invisibles
  base04: "#969896", // dark foreground
  base05: "#d9d9d9", // default foreground
  base06: "#e8e8e8", // light foreground
  base07: "#ffffff", // light background
  base08: "#ff5370", // variables, red
  base09: "#f78c6c", // integers, orange
  base0A: "#ffcb6b", // booleans, yellow
  base0B: "#c3e88d", // strings, green
  base0C: "#89ddff", // support, cyan
  base0D: "#82aaff", // functions, blue
  base0E: "#c792ea", // keywords, purple
  base0F: "#ff5370", // deprecated, red
};

export const DOCUMENTATION_URL = {
  MICROAGENTS: {
    MICROAGENTS_OVERVIEW:
      "https://docs.all-hands.dev/usage/prompting/microagents-overview",
    ORGANIZATION_AND_USER_MICROAGENTS:
      "https://docs.all-hands.dev/usage/prompting/microagents-org",
  },
};

export const PRODUCT_URL = {
  PRODUCTION: "https://app.all-hands.dev",
};

export const SETTINGS_FORM = {
  LABEL_CLASSNAME: "text-[11px] font-medium leading-4 tracking-[0.11px]",
};

export const GIT_PROVIDER_OPTIONS = [
  {
    label: "GitHub",
    value: "github",
  },
  {
    label: "GitLab",
    value: "gitlab",
  },
  {
    label: "Bitbucket",
    value: "bitbucket",
  },
];

export const CONTEXT_MENU_ICON_TEXT_CLASSNAME = "h-[30px]";

// Chat input constants
export const CHAT_INPUT = {
  HEIGHT_THRESHOLD: 100, // Height in pixels when suggestions should be hidden
};

// UI tolerance constants
export const EPS = 1.5; // px tolerance for "near min" height comparisons
