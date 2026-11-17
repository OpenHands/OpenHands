import "i18next";

// i18next 27 enables `strictInterpolationChecks` by default in its types, which
// makes the `t()` options required for any key whose translation contains a
// `{{variable}}`. The app has no `CustomTypeOptions.resources`, so keys are
// typed as `string` and the check cannot narrow to the affected keys: it rejects
// every call that forwards an optional options object, such as the shared
// `(key, options?) => string` wrappers (`formatRelativeTime`, `Translate`, ...)
// and the `t` returned by `useTranslation`. Restore the v26 types, which the
// migration guide documents as the supported escape hatch.
declare module "i18next" {
  interface CustomTypeOptions {
    strictInterpolationChecks: false;
  }
}
