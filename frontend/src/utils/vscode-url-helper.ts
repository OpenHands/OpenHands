/**
 * Helper function to transform VS Code URLs
 *
 * This function checks if a VS Code URL points to localhost and replaces it with
 * the current window's hostname if they don't match.
 *
 * @param vsCodeUrl The original VS Code URL from the backend
 * @returns The transformed URL with the correct hostname
 */
export function transformVSCodeUrl(vsCodeUrl: string | null): string | null {
  if (!vsCodeUrl) return null;

  try {
    const url = new URL(vsCodeUrl);

    // Check if the URL points to localhost
    if (
      url.hostname === "localhost" &&
      window.location.hostname !== "localhost"
    ) {
      // Replace localhost with the current hostname
      url.hostname = window.location.hostname;
      return url.toString();
    }

    return vsCodeUrl;
  } catch {
    // Silently handle the error and return the original URL
    return vsCodeUrl;
  }
}

/**
 * Constructs a VSCode Remote SSH URI for opening the sandbox in local VSCode
 *
 * The URI format is: vscode://vscode-remote/ssh-remote+[user@]hostname[:port]/path/to/folder
 * This opens VS Code with the Remote-SSH extension and connects to the specified host.
 *
 * @param host The SSH host (hostname or IP)
 * @param port The SSH port
 * @param folderPath The folder path to open on the remote
 * @param username The SSH username (default: 'openhands')
 * @param newWindow Whether to request opening in a new window (default: true)
 * @returns The VSCode Remote SSH URI
 */
export function buildVSCodeRemoteSSHUrl(
  host: string,
  port: number,
  folderPath: string,
  username: string = "openhands",
  newWindow: boolean = true,
): string {
  // VSCode Remote SSH URI format: vscode://vscode-remote/ssh-remote+user@host:port/path
  const sshTarget = `${username}@${host}:${port}`;
  const baseUrl = `vscode://vscode-remote/ssh-remote+${sshTarget}${folderPath}`;
  // Add windowId=_blank to request a new window (supported in recent VSCode versions)
  return newWindow ? `${baseUrl}?windowId=_blank` : baseUrl;
}

/**
 * Extracts SSH connection info from an exposed URL
 *
 * @param sshUrl The SSH URL from sandbox exposed_urls (e.g., "http://localhost:12345")
 * @returns Object with host and port, or null if invalid
 */
export function extractSSHConnectionInfo(
  sshUrl: string | null,
): { host: string; port: number } | null {
  if (!sshUrl) return null;

  try {
    const url = new URL(sshUrl);
    const host =
      url.hostname === "localhost" && window.location.hostname !== "localhost"
        ? window.location.hostname
        : url.hostname;
    const port = parseInt(url.port, 10);

    if (Number.isNaN(port)) return null;

    return { host, port };
  } catch {
    return null;
  }
}
