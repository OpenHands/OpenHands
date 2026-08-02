import { useCallback, useEffect, useRef } from "react";
import type {
  BashCommand,
  BashError,
  BashEvent,
  BashOutput,
} from "@openhands/typescript-client";
import type { CommandResult } from "#/api/runtime-service/agent-server-runtime-service";
import { sendWebSocketAuth } from "#/utils/websocket-auth";
import { buildBashWebSocketUrl } from "#/utils/websocket-url";

interface WaitingCommand {
  command: string;
  cwd: string;
  timeout: number;
  resolve: (result: CommandResult) => void;
  reject: (error: Error) => void;
}

interface PendingCommand {
  resolve: (result: CommandResult) => void;
  reject: (error: Error) => void;
}

interface ActiveCommand extends PendingCommand {
  stdout: string[];
  stderr: string[];
}

export type BashCommandRunner = (
  command: string,
  cwd: string,
  timeout: number,
) => Promise<CommandResult>;

function isBashCommand(event: BashEvent): event is BashCommand {
  return event.kind === "BashCommand";
}

function isBashOutput(event: BashEvent): event is BashOutput {
  return event.kind === "BashOutput";
}

function isBashError(event: BashEvent): event is BashError {
  return event.kind === "BashError";
}

/**
 * Maintains a persistent WebSocket connection to the agent-server's
 * `/sockets/bash-events` endpoint and exposes a `runCommand` function that
 * executes a bash command and returns a Promise that resolves when the
 * final `BashOutput` (non-null `exit_code`) arrives.
 *
 * Commands are correlated using a FIFO queue: each `BashCommand` echo
 * received from the server is paired with the oldest outstanding request in
 * the queue, and subsequent `BashOutput` events are matched by `command_id`.
 *
 * Commands are buffered until the socket's open handler sends authentication.
 */
export function useBashCommandRunner(
  conversationUrl: string | null | undefined,
  sessionApiKey: string | null | undefined,
  enabled: boolean,
): BashCommandRunner {
  const wsRef = useRef<WebSocket | null>(null);
  const readyWsRef = useRef<WebSocket | null>(null);
  const waitingQueueRef = useRef<WaitingCommand[]>([]);
  // Commands whose request was sent; waiting for the BashCommand echo to get command_id
  const pendingQueueRef = useRef<PendingCommand[]>([]);
  // Commands whose command_id is known; waiting for BashOutput with non-null exit_code
  const activeCommandsRef = useRef<Map<string, ActiveCommand>>(new Map());

  useEffect(() => {
    if (!enabled) return;

    const wsUrl = buildBashWebSocketUrl(conversationUrl);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    readyWsRef.current = null;

    ws.onopen = () => {
      sendWebSocketAuth(ws, sessionApiKey);
      readyWsRef.current = ws;
      for (const {
        command,
        cwd,
        timeout,
        resolve,
        reject,
      } of waitingQueueRef.current) {
        pendingQueueRef.current.push({ resolve, reject });
        ws.send(JSON.stringify({ command, cwd, timeout }));
      }
      waitingQueueRef.current = [];
    };

    ws.onmessage = (event: MessageEvent) => {
      let data: BashEvent;
      try {
        data = JSON.parse(event.data as string) as BashEvent;
      } catch {
        return; // ignore malformed frames
      }

      if (isBashCommand(data)) {
        // Associate the next pending request with the server-assigned command_id
        const pending = pendingQueueRef.current.shift();
        if (pending) {
          activeCommandsRef.current.set(data.id, {
            ...pending,
            stdout: [],
            stderr: [],
          });
        }
      } else if (isBashOutput(data) && data.command_id) {
        const active = activeCommandsRef.current.get(data.command_id);
        if (active) {
          if (data.stdout) active.stdout.push(data.stdout);
          if (data.stderr) active.stderr.push(data.stderr);
          if (data.exit_code != null) {
            activeCommandsRef.current.delete(data.command_id);
            active.resolve({
              exit_code: data.exit_code,
              stdout: active.stdout.join(""),
              stderr: active.stderr.join(""),
            });
          }
        }
      } else if (isBashError(data)) {
        rejectAll(`Bash error: ${data.code}: ${data.detail}`);
      }
    };

    function rejectAll(reason: string): void {
      const err = new Error(reason);
      for (const { reject: rej } of waitingQueueRef.current) rej(err);
      waitingQueueRef.current = [];
      for (const p of pendingQueueRef.current) p.reject(err);
      pendingQueueRef.current = [];
      for (const a of activeCommandsRef.current.values()) a.reject(err);
      activeCommandsRef.current.clear();
    }

    ws.onclose = () => {
      wsRef.current = null;
      readyWsRef.current = null;
      rejectAll("Bash WebSocket closed");
    };

    ws.onerror = () => {
      wsRef.current = null;
      readyWsRef.current = null;
      rejectAll("Bash WebSocket error");
    };

    return () => {
      // Prevent the close/error handlers from double-rejecting after unmount
      ws.onclose = null;
      ws.onerror = null;
      ws.close();
      wsRef.current = null;
      readyWsRef.current = null;
      rejectAll("Bash WebSocket unmounted");
    };
  }, [enabled, conversationUrl, sessionApiKey]);

  const runCommand: BashCommandRunner = useCallback(
    (command: string, cwd: string, timeout: number) =>
      new Promise<CommandResult>((resolve, reject) => {
        const ws = wsRef.current;
        if (
          !ws ||
          ws.readyState === WebSocket.CLOSED ||
          ws.readyState === WebSocket.CLOSING
        ) {
          reject(new Error("Bash WebSocket not available"));
          return;
        }
        if (ws.readyState !== WebSocket.OPEN || readyWsRef.current !== ws) {
          waitingQueueRef.current.push({
            command,
            cwd,
            timeout,
            resolve,
            reject,
          });
        } else {
          pendingQueueRef.current.push({ resolve, reject });
          ws.send(JSON.stringify({ command, cwd, timeout }));
        }
      }),
    [],
  );

  return runCommand;
}
