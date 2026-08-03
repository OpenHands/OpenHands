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
  // Kept so a `BashCommand` echo can be matched back to the request that
  // produced it — see `takeMatchingPending`.
  command: string;
  cwd: string;
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
 * Remove and return the queued request that a `BashCommand` echo belongs to,
 * or `null` when the echo is not ours.
 *
 * The agent-server subscribes every `/sockets/bash-events` connection to one
 * shared `BashEventService`, and `start_bash_command` publishes to all
 * subscribers regardless of origin — including commands started through
 * `POST /api/bash/start_bash_command` by an automation run, another tab, or
 * the SDK. So an echo arriving here is *not* necessarily a reply to us, and
 * pairing echoes with the oldest outstanding request by position alone let a
 * foreign command capture our promise and resolve it with its own output
 * (a filename would surface as the git branch — #15543).
 *
 * `command` is the match key: it is the only field the protocol marks as
 * required (`BashCommand extends ExecuteBashRequest`) and the server echoes it
 * verbatim, since it is the literal string it has to execute. `cwd` is a
 * tiebreaker rather than part of the key — it is optional in the protocol, so
 * requiring it to match would strand the promise forever against any server
 * that omitted it. The tiebreaker matters when two conversations probe
 * different workspaces with the same script: without it, one could adopt the
 * other's result.
 *
 * Two of *our own* in-flight requests sharing a command and cwd are
 * interchangeable, so taking the oldest is correct. A foreign command that
 * happens to be byte-identical to ours (a second tab running the same git
 * probe) can still be adopted, which is harmless: identical command, identical
 * cwd, equivalent output.
 */
function takeMatchingPending(
  queue: PendingCommand[],
  echo: BashCommand,
): PendingCommand | null {
  const matchesCommand = (pending: PendingCommand) =>
    pending.command === echo.command;

  let index = queue.findIndex(
    (pending) => matchesCommand(pending) && pending.cwd === echo.cwd,
  );
  if (index === -1) {
    index = queue.findIndex(matchesCommand);
  }
  if (index === -1) {
    return null;
  }

  const [pending] = queue.splice(index, 1);
  return pending;
}

/**
 * Maintains a persistent WebSocket connection to the agent-server's
 * `/sockets/bash-events` endpoint and exposes a `runCommand` function that
 * executes a bash command and returns a Promise that resolves when the
 * final `BashOutput` (non-null `exit_code`) arrives.
 *
 * The socket is a shared broadcast, not a private channel, so each
 * `BashCommand` echo is matched back to the request that produced it by
 * command text (see `takeMatchingPending`) rather than by queue position;
 * subsequent `BashOutput` events are then matched by `command_id`.
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
        pendingQueueRef.current.push({ command, cwd, resolve, reject });
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
        // Associate the matching pending request with the server-assigned
        // command_id. Echoes for commands we did not send are ignored.
        const pending = takeMatchingPending(pendingQueueRef.current, data);
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
          pendingQueueRef.current.push({ command, cwd, resolve, reject });
          ws.send(JSON.stringify({ command, cwd, timeout }));
        }
      }),
    [],
  );

  return runCommand;
}
