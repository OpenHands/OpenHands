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
      },
    },
  },
  plugins: [typography],
};
