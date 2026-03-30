import { heroui } from "@heroui/react";

export default heroui({
  defaultTheme: "dark",
  layout: {
    radius: {
      small: "5px",
      large: "20px",
    },
  },
  themes: {
    dark: {
      colors: {
        primary: "#c084fc",
      },
    },
    light: {
      colors: {
        primary: "#9333ea",
      },
    },
  },
});
