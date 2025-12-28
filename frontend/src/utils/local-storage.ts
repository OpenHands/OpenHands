// Local storage keys
export const LOCAL_STORAGE_KEYS = {
  LOGIN_METHOD: "openhands_login_method",
};

// Login methods
export enum LoginMethod {
  GITHUB = "github",
  GITLAB = "gitlab",
  BITBUCKET = "bitbucket",
  AZURE_DEVOPS = "azure_devops",
  ENTERPRISE_SSO = "enterprise_sso",
}

/**
 * Set the login method in local storage
 * @param method The login method (github, gitlab, bitbucket, or azure_devops)
 */
export const setLoginMethod = (method: LoginMethod): void => {
  localStorage.setItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD, method);
};

/**
 * Get the login method from local storage
 * @returns The login method or null if not set
 */
export const getLoginMethod = (): LoginMethod | null => {
  const method = localStorage.getItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD);
  return method as LoginMethod | null;
};

/**
 * Clear login method and last page from local storage
 */
export const clearLoginData = (): void => {
  localStorage.removeItem(LOCAL_STORAGE_KEYS.LOGIN_METHOD);
};
