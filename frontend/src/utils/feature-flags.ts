export function loadFeatureFlag(
  flagName: string,
  defaultValue: boolean = false,
): boolean {
  try {
    const stringValue =
      localStorage.getItem(`FEATURE_${flagName}`) || defaultValue.toString();
    const value = !!JSON.parse(stringValue);
    return value;
  } catch (e) {
    return defaultValue;
  }
}

export const BILLING_SETTINGS = () => loadFeatureFlag("BILLING_SETTINGS");
export const HIDE_LLM_SETTINGS = () => loadFeatureFlag("HIDE_LLM_SETTINGS");
export const VSCODE_IN_NEW_TAB = () => loadFeatureFlag("VSCODE_IN_NEW_TAB");
export const ENABLE_TRAJECTORY_REPLAY = () =>
  loadFeatureFlag("TRAJECTORY_REPLAY");
