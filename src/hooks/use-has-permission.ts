// Simple permission hook for OSS agent-canvas
// In the OSS context, all authenticated users have full access
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function useHasPermission(permission: string): boolean {
  // In OSS mode, every permission string is granted.
  return true;
}
