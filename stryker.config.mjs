// @ts-check

/** @type {import("@stryker-mutator/api/core").PartialStrykerOptions} */
const config = {
  testRunner: "vitest",
  mutate: [
    "src/**/*.{ts,tsx}",
    "!src/**/*.{test,spec}.{ts,tsx}",
    "!src/**/__tests__/**",
    "!src/**/*.d.ts",
    "!src/**/*.types.ts",
    "!src/**/*.{gen,generated}.{ts,tsx}",
    "!src/i18n/declaration.ts",
    "!src/{fixtures,mocks,dev}/**",
  ],
  vitest: {
    configFile: "vite.config.ts",
    related: true,
  },
};

export default config;
