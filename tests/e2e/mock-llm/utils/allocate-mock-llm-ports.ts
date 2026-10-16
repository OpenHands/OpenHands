import {
  connect as connectLease,
  createServer,
  type AddressInfo,
  type Server,
} from "node:net";
import { fileURLToPath, pathToFileURL } from "node:url";

/**
 * Env var holding the ports chosen by the first evaluation of a mock-LLM
 * Playwright config. Workers re-import that config after webServers bind, so
 * a second findFreePorts() pass treats the live ports as busy and points
 * baseURL / MOCK_LLM_BACKEND_URL at a different set. The snapshot is copied
 * into the worker via process.env at fork time. A file under RUN_ROOT would
 * not survive: the webServer command deletes RUN_ROOT before workers start.
 */
export const MOCK_LLM_PORT_SNAPSHOT_ENV = "MOCK_LLM_PORT_SNAPSHOT";

/**
 * `host:port` of the runner process that holds every allocated port.
 * WebServer commands and workers inherit it and ask the runner to release a
 * port just before the process that binds it starts.
 */
export const MOCK_LLM_PORT_LEASE_ENV = "MOCK_LLM_PORT_LEASE";

/** `dev-with-automation.mjs` binds vscode at `backendPort + 1000`. */
const VSCODE_OFFSET = 1000;
/** Ingress, backend (+1), automation (+2), and vite (+3). */
const CHILD_STACK_SPAN = 4;
/** Child backend is `base + 1`, so its vscode port is `base + 1001`. */
const CHILD_STACK_VSCODE_OFFSET = 1 + VSCODE_OFFSET;

export type MockLlmPortMode = "npm" | "docker";

export type MockLlmPortSpec = {
  name: string;
  envName: string;
  /** Explicit port to try first; 0 lets the OS choose. */
  preferred: number;
  /** Contiguous ports reserved from the base. Defaults to 1. */
  span?: number;
  /** Extra offsets from the base, held with the contiguous span. */
  extraOffsets?: number[];
  /**
   * Keep the sockets open in this process so a concurrent run sees them as
   * busy until the owner releases them right before binding.
   */
  hold?: boolean;
};

type PortSnapshot = {
  scope: string;
  ports: Record<string, number>;
};

type HeldBlock = {
  name: string;
  ports: number[];
  servers: Server[];
};

type LeaseRequest = {
  op: "release" | "rehold" | "shutdown";
  names?: string[];
};

type LeaseResponse = { ok: true } | { ok: false; error: string };

const heldBlocks: HeldBlock[] = [];
let leaseServer: Server | undefined;
let leaseAddress: string | undefined;

function preferredPort(env: NodeJS.ProcessEnv, names: string[]): number {
  for (const name of names) {
    const raw = env[name]?.trim();
    if (!raw) continue;
    const port = Number(raw);
    if (Number.isInteger(port) && port > 0 && port <= 65535) return port;
  }
  return 0;
}

function serverPort(
  env: NodeJS.ProcessEnv,
  name: string,
  envNames: string[],
  extraOffsets: number[] = [],
): MockLlmPortSpec {
  return {
    name,
    envName: envNames[0],
    preferred: preferredPort(env, envNames),
    extraOffsets,
    hold: true,
  };
}

function childStack(
  env: NodeJS.ProcessEnv,
  name: string,
  envName: string,
): MockLlmPortSpec {
  return {
    name,
    envName,
    preferred: preferredPort(env, [envName]),
    span: CHILD_STACK_SPAN,
    extraOffsets: [CHILD_STACK_VSCODE_OFFSET],
    hold: true,
  };
}

/**
 * Port list shared by the npm and Docker mock-LLM configs.
 * There are no fixed defaults: two runs that both preferred 18300 would
 * still race once the first released it for its server. Every port is held
 * until the process that binds it starts; see `releasePortsCommand`.
 * Child stacks reserve the four ports `createIsolatedEnv` derives from the
 * base, plus the vscode port, and the runner holds them until the spec binds.
 */
export function mockLlmPortSpecs(
  mode: MockLlmPortMode,
  env: NodeJS.ProcessEnv = process.env,
): MockLlmPortSpec[] {
  const specs: MockLlmPortSpec[] = [
    serverPort(env, "mockLlm", ["MOCK_LLM_PORT"]),
    serverPort(
      env,
      "ingress",
      mode === "npm"
        ? ["MOCK_LLM_INGRESS_PORT", "PORT"]
        : ["MOCK_LLM_INGRESS_PORT"],
    ),
    serverPort(env, "publicMode", ["MOCK_LLM_PUBLIC_MODE_PORT"]),
    serverPort(
      env,
      "backend",
      mode === "docker"
        ? ["OH_CANVAS_SAFE_BACKEND_PORT", "AGENT_SERVER_PORT"]
        : ["OH_CANVAS_SAFE_BACKEND_PORT"],
      mode === "npm" ? [VSCODE_OFFSET] : [],
    ),
    serverPort(
      env,
      "automation",
      mode === "docker"
        ? ["OH_CANVAS_SAFE_AUTOMATION_PORT", "AUTOMATION_PORT"]
        : ["OH_CANVAS_SAFE_AUTOMATION_PORT"],
    ),
  ];
  if (mode === "npm") {
    specs.push(serverPort(env, "vite", ["OH_CANVAS_SAFE_VITE_PORT"]));
  } else {
    // The image binds a fixed vscode port (8001) unless told otherwise, and
    // `--network host` would make two containers collide on it.
    specs.push(serverPort(env, "vscode", ["MOCK_LLM_VSCODE_PORT"]));
  }
  specs.push(
    childStack(env, "feOnly", "MOCK_LLM_FE_ONLY_PORT"),
    childStack(env, "beOnly", "MOCK_LLM_BE_ONLY_PORT"),
    childStack(env, "conflict", "MOCK_LLM_CONFLICT_PORT"),
    childStack(env, "conflictFree", "MOCK_LLM_CONFLICT_FREE_PORT"),
    childStack(env, "crossFe", "MOCK_LLM_CROSS_FE_PORT"),
    childStack(env, "crossBeA", "MOCK_LLM_CROSS_BE_A_PORT"),
    childStack(env, "crossBeB", "MOCK_LLM_CROSS_BE_B_PORT"),
  );
  return specs;
}

function blockPorts(
  base: number,
  span: number,
  extraOffsets: number[],
): number[] {
  const ports: number[] = [];
  for (let offset = 0; offset < span; offset += 1) ports.push(base + offset);
  for (const offset of extraOffsets) ports.push(base + offset);
  return ports;
}

function tryBind(port: number): Promise<Server | null> {
  return new Promise((resolve, reject) => {
    // Reset connections so Playwright's pre-launch URL probe fails fast
    // instead of waiting for an HTTP reply that never comes.
    const server = createServer((socket) => socket.destroy());
    const onError = (err: NodeJS.ErrnoException) => {
      server.unref();
      if (err.code === "EADDRINUSE" || err.code === "EACCES") resolve(null);
      else reject(err);
    };
    server.once("error", onError);
    server.listen(port, "127.0.0.1", () => {
      server.removeListener("error", onError);
      server.unref();
      resolve(server);
    });
  });
}

function closeServer(server: Server): Promise<void> {
  return new Promise((resolve) => {
    if (!server.listening) {
      resolve();
      return;
    }
    server.close(() => resolve());
  });
}

async function bindEphemeral(): Promise<{ port: number; server: Server }> {
  const server = await tryBind(0);
  if (!server) throw new Error("could not bind an ephemeral port");
  const address = server.address();
  if (!address || typeof address === "string") {
    await closeServer(server);
    throw new Error("ephemeral port did not expose a numeric address");
  }
  return { port: address.port, server };
}

async function allocateSingle(
  preferred: number,
  used: Set<number>,
  hold: boolean,
): Promise<{ base: number; ports: number[]; servers: Server[] }> {
  if (preferred > 0 && !used.has(preferred)) {
    const server = await tryBind(preferred);
    if (server) {
      if (!hold) await closeServer(server);
      return {
        base: preferred,
        ports: [preferred],
        servers: hold ? [server] : [],
      };
    }
  }

  for (let attempt = 0; attempt < 100; attempt += 1) {
    const bound = await bindEphemeral();
    if (used.has(bound.port)) {
      await closeServer(bound.server);
      continue;
    }
    if (!hold) await closeServer(bound.server);
    return {
      base: bound.port,
      ports: [bound.port],
      servers: hold ? [bound.server] : [],
    };
  }
  throw new Error("could not allocate a unique port");
}

function blockCandidates(preferred: number, span: number): number[] {
  const start =
    preferred > 0 ? preferred : 20_000 + Math.floor(Math.random() * 500) * span;
  const candidates = [start];
  for (let step = 1; step <= 400; step += 1) {
    const next = start + step * span;
    if (next > 65_000) break;
    candidates.push(next);
  }
  return candidates;
}

async function allocateBlock(
  preferred: number,
  span: number,
  extraOffsets: number[],
  used: Set<number>,
  hold: boolean,
): Promise<{ base: number; ports: number[]; servers: Server[] }> {
  for (const base of blockCandidates(preferred, span)) {
    const ports = blockPorts(base, span, extraOffsets);
    if (ports.some((port) => port < 1024 || port > 65535 || used.has(port))) {
      continue;
    }
    const servers: Server[] = [];
    let failed = false;
    for (const port of ports) {
      const server = await tryBind(port);
      if (!server) {
        failed = true;
        break;
      }
      servers.push(server);
    }
    if (failed) {
      await Promise.all(servers.map((server) => closeServer(server)));
      continue;
    }
    if (!hold) await Promise.all(servers.map((server) => closeServer(server)));
    return { base, ports, servers: hold ? servers : [] };
  }
  throw new Error(`could not allocate a free ${span}-port block`);
}

function readPortSnapshot(raw: string | undefined): PortSnapshot | undefined {
  const trimmed = raw?.trim();
  if (!trimmed) return undefined;

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    throw new Error(`${MOCK_LLM_PORT_SNAPSHOT_ENV} is not valid JSON`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${MOCK_LLM_PORT_SNAPSHOT_ENV} must be a JSON object`);
  }

  const record = parsed as { scope?: unknown; ports?: unknown };
  if (typeof record.scope !== "string" || !record.scope) {
    throw new Error(`${MOCK_LLM_PORT_SNAPSHOT_ENV} is missing a scope`);
  }
  if (
    !record.ports ||
    typeof record.ports !== "object" ||
    Array.isArray(record.ports)
  ) {
    throw new Error(`${MOCK_LLM_PORT_SNAPSHOT_ENV} is missing ports`);
  }

  const ports: Record<string, number> = {};
  for (const [name, value] of Object.entries(record.ports)) {
    const port = typeof value === "number" ? value : Number(value);
    if (!Number.isInteger(port) || port <= 0 || port > 65535) {
      throw new Error(
        `${MOCK_LLM_PORT_SNAPSHOT_ENV} has an invalid port for "${name}"`,
      );
    }
    ports[name] = port;
  }
  return { scope: record.scope, ports };
}

function publishLease(env: NodeJS.ProcessEnv) {
  if (!leaseAddress) return;
  Object.assign(env, { [MOCK_LLM_PORT_LEASE_ENV]: leaseAddress });
}

function applySnapshot<Name extends string>(
  specs: Array<{ name: Name; envName: string }>,
  snapshot: Record<string, number>,
  env: NodeJS.ProcessEnv,
): Record<Name, string> {
  const ports = {} as Record<Name, string>;
  const updates: Record<string, string> = {};
  for (const spec of specs) {
    const value = String(snapshot[spec.name]);
    updates[spec.envName] = value;
    ports[spec.name] = value;
  }
  Object.assign(env, updates);
  publishLease(env);
  return ports;
}

function startLeaseServer(env: NodeJS.ProcessEnv): Promise<void> {
  if (leaseServer && leaseAddress) {
    publishLease(env);
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const server = createServer((socket) => {
      socket.unref();
      let buf = "";
      socket.on("data", (chunk) => {
        buf += chunk.toString();
        const newline = buf.indexOf("\n");
        if (newline === -1) return;
        const line = buf.slice(0, newline);
        buf = buf.slice(newline + 1);
        const shutdown = line.includes('"op":"shutdown"');
        void handleLeaseMessage(line)
          .then((response) => {
            socket.end(`${JSON.stringify(response)}\n`);
            if (shutdown && response.ok) {
              socket.once("finish", () => {
                void closeLease();
              });
            }
          })
          .catch((err: unknown) => {
            const error = err instanceof Error ? err.message : String(err);
            socket.end(`${JSON.stringify({ ok: false, error })}\n`);
          });
      });
    });
    server.unref();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address() as AddressInfo;
      leaseAddress = `127.0.0.1:${address.port}`;
      leaseServer = server;
      publishLease(env);
      resolve();
    });
  });
}

async function handleLeaseMessage(line: string): Promise<LeaseResponse> {
  let message: LeaseRequest;
  try {
    message = JSON.parse(line) as LeaseRequest;
  } catch {
    return { ok: false, error: "port lease request is not valid JSON" };
  }
  if (message.op === "shutdown") return { ok: true };
  const names = message.names ?? [];
  if (message.op === "release") {
    for (const name of names) await releaseLocal(name);
    return { ok: true };
  }
  if (message.op === "rehold") {
    for (const name of names) await reholdLocal(name);
    return { ok: true };
  }
  return { ok: false, error: "unknown port lease request" };
}

async function releaseLocal(name: string): Promise<void> {
  for (const block of heldBlocks) {
    if (block.name !== name || block.servers.length === 0) continue;
    const servers = block.servers.splice(0);
    await Promise.all(servers.map((server) => closeServer(server)));
  }
}

async function reholdLocal(name: string): Promise<void> {
  for (const block of heldBlocks) {
    if (block.name !== name || block.servers.length > 0) continue;
    const servers: Server[] = [];
    for (let attempt = 0; attempt < 20; attempt += 1) {
      servers.splice(0);
      let failed = false;
      for (const port of block.ports) {
        const server = await tryBind(port);
        if (!server) {
          failed = true;
          break;
        }
        servers.push(server);
      }
      if (!failed) {
        block.servers = servers;
        return;
      }
      await Promise.all(servers.map((server) => closeServer(server)));
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    throw new Error(
      `could not re-hold ports for "${name}": ${block.ports.join(", ")}`,
    );
  }
}

function sendLease(
  env: NodeJS.ProcessEnv,
  message: LeaseRequest,
): Promise<void> {
  const address = env[MOCK_LLM_PORT_LEASE_ENV]?.trim() || leaseAddress;
  if (!address) return Promise.resolve();
  const separator = address.lastIndexOf(":");
  const host = address.slice(0, separator);
  const port = Number(address.slice(separator + 1));
  return new Promise((resolve, reject) => {
    const socket = connectLease({ host, port });
    let buf = "";
    socket.on("error", reject);
    socket.on("connect", () => {
      socket.write(`${JSON.stringify(message)}\n`);
    });
    socket.on("data", (chunk) => {
      buf += chunk.toString();
      if (!buf.includes("\n")) return;
      socket.end();
      try {
        const response = JSON.parse(buf) as LeaseResponse;
        if (!response.ok) reject(new Error(response.error));
        else resolve();
      } catch (err) {
        reject(err);
      }
    });
  });
}

function localBlocks(names: string[]): boolean {
  return names.some((name) => heldBlocks.some((block) => block.name === name));
}

/**
 * Close held sockets so the spec's child stack can bind them.
 * No-op when this process never started a lease (specs run outside Playwright).
 */
export async function releaseHeldPorts(
  names: string[],
  env: NodeJS.ProcessEnv = process.env,
): Promise<void> {
  if (localBlocks(names)) {
    for (const name of names) await releaseLocal(name);
    return;
  }
  await sendLease(env, { op: "release", names });
}

/** Bind a block again after its child stack exits. */
export async function reholdPorts(
  names: string[],
  env: NodeJS.ProcessEnv = process.env,
): Promise<void> {
  if (localBlocks(names)) {
    for (const name of names) await reholdLocal(name);
    return;
  }
  await sendLease(env, { op: "rehold", names });
}

async function closeLease(): Promise<void> {
  const blocks = heldBlocks.splice(0);
  await Promise.all(
    blocks.flatMap((block) =>
      block.servers.map((server) => closeServer(server)),
    ),
  );
  const server = leaseServer;
  leaseServer = undefined;
  leaseAddress = undefined;
  if (!server) return;
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

/** Drop every held port. Used by teardown and unit tests. */
export async function shutdownPortLease(
  env: NodeJS.ProcessEnv = process.env,
): Promise<void> {
  if (leaseServer || heldBlocks.length > 0) {
    await closeLease();
    return;
  }
  try {
    await sendLease(env, { op: "shutdown" });
  } catch (err) {
    console.warn("Failed to release mock-LLM port lease:", err);
  }
}

/**
 * Allocate each port once per process tree. Later evaluations, including
 * Playwright workers, reuse the snapshot instead of probing again.
 * Explicit preferred values apply only to the first allocation.
 * Held child-stack blocks stay bound in the runner until a spec releases them.
 */
export async function allocateMockLlmPorts<Name extends string>(
  specs: Array<MockLlmPortSpec & { name: Name }>,
  scope: string,
  env: NodeJS.ProcessEnv = process.env,
): Promise<Record<Name, string>> {
  const snapshot = readPortSnapshot(env[MOCK_LLM_PORT_SNAPSHOT_ENV]);
  if (snapshot?.scope === scope) {
    const missing = specs.filter(
      (spec) => snapshot.ports[spec.name] === undefined,
    );
    if (missing.length > 0) {
      throw new Error(
        `${MOCK_LLM_PORT_SNAPSHOT_ENV} is missing ${missing
          .map((spec) => spec.name)
          .join(", ")}`,
      );
    }
    return applySnapshot(specs, snapshot.ports, env);
  }

  const chosen: Record<string, number> = {};
  const used = new Set<number>();
  for (const spec of specs) {
    const span = spec.span ?? 1;
    const extraOffsets = spec.extraOffsets ?? [];
    const hold = spec.hold === true;
    const allocated =
      span === 1 && extraOffsets.length === 0
        ? await allocateSingle(spec.preferred, used, hold)
        : await allocateBlock(spec.preferred, span, extraOffsets, used, hold);
    chosen[spec.name] = allocated.base;
    for (const port of allocated.ports) used.add(port);
    if (allocated.servers.length > 0) {
      await startLeaseServer(env);
      heldBlocks.push({
        name: spec.name,
        ports: allocated.ports,
        servers: allocated.servers,
      });
    }
  }
  publishLease(env);
  Object.assign(env, {
    [MOCK_LLM_PORT_SNAPSHOT_ENV]: JSON.stringify({ scope, ports: chosen }),
  });
  return applySnapshot(specs, chosen, env);
}

const MODULE_PATH = fileURLToPath(import.meta.url);

/**
 * Shell prefix for a webServer command: asks the runner to release the named
 * held ports, then runs the rest of the command, which binds them.
 */
export function releasePortsCommand(names: string[]): string {
  const modulePath = `'${MODULE_PATH.replaceAll("'", "'\\''")}'`;
  return `node ${modulePath} release ${names.join(" ")} && `;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href &&
  process.argv[2] === "release"
) {
  releaseHeldPorts(process.argv.slice(3)).catch((err: unknown) => {
    console.error("Failed to release mock-LLM ports:", err);
    process.exitCode = 1;
  });
}
