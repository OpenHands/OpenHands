/** @type {import('tailwindcss').Config} */
import { heroui } from "@heroui/react";
import typography from "@tailwindcss/typography";
export default {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        modal: {
          background: "#171717",
          input: "#27272A",
          primary: "#F3CE49",
          secondary: "#737373",
          muted: "#A3A3A3",
        },
        org: {
          border: "#171717",
          background: "#262626",
          divider: "#525252",
          button: "#737373",
          text: "#A3A3A3",
        },
      },
    },
  },
  plugins: [typography],
};
