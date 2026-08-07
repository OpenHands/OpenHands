/**
 * Stable secret names for GitProviders clone credentials.
 * Must stay in sync with agent-server `workspace_clone.py`.
 */
export function gitProviderTokenSecretName(providerId: string): string {
  return `GIT_PROVIDER_${providerId}_TOKEN`;
}

export function gitProviderUsernameSecretName(providerId: string): string {
  return `GIT_PROVIDER_${providerId}_USERNAME`;
}

export function gitProviderPasswordSecretName(providerId: string): string {
  return `GIT_PROVIDER_${providerId}_PASSWORD`;
}

export function gitProviderSshPrivateKeySecretName(providerId: string): string {
  return `GIT_PROVIDER_${providerId}_SSH_PRIVATE_KEY`;
}

export function allGitProviderSecretNames(providerId: string): string[] {
  return [
    gitProviderTokenSecretName(providerId),
    gitProviderUsernameSecretName(providerId),
    gitProviderPasswordSecretName(providerId),
    gitProviderSshPrivateKeySecretName(providerId),
  ];
}
