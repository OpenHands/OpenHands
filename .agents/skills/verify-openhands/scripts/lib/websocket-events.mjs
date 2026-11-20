// Subscription metadata only: payloads and headers never enter the event log.
import { redactUrl } from "./network-bodies.mjs";

export function webSocketMetadata(raw) {
  try {
    const url = new URL(raw);
    if (url.href === "about:blank") return { url: url.href };
    if (!["http:", "https:", "ws:", "wss:"].includes(url.protocol))
      return { url: "<unavailable>" };
    url.username = "";
    url.password = "";
    url.hash = "";
    const cursors = url.searchParams.getAll("after_seq");
    const cursor = cursors.length === 1 ? cursors[0] : undefined;
    const afterSeq = /^-?\d+$/.test(cursor ?? "") ? cursor : undefined;
    // Allow only one numeric cursor. Even unknown parameter names/values
    // may contain credentials, so omit the rest rather than guessing.
    const redactedQueryParameters =
      [...url.searchParams].length - (afterSeq === undefined ? 0 : 1);
    url.search = "";
    if (afterSeq !== undefined) url.searchParams.set("after_seq", afterSeq);
    return {
      url: redactUrl(url.toString()),
      ...(afterSeq === undefined ? {} : { afterSeq }),
      ...(redactedQueryParameters ? { redactedQueryParameters } : {}),
    };
  } catch {
    return { url: "<unavailable>" };
  }
}

export function isWebSocketObservation(kind) {
  return ["websocket-created", "websocket-open", "websocket-close"].includes(
    kind,
  );
}

export function observeWebSocket(socket, record, { connectionId, pageUrl }) {
  const metadata = webSocketMetadata(socket.url());
  let receivedFrames = 0;
  let sentFrames = 0;
  let observedOpen = false;
  const emit = (kind, detail = {}) =>
    record(kind, {
      connectionId,
      ...metadata,
      page: webSocketMetadata(pageUrl()).url,
      receivedFrames,
      sentFrames,
      ...detail,
    });
  emit("websocket-created");
  const frame = (received) => {
    if (received) receivedFrames += 1;
    else sentFrames += 1;
    if (!observedOpen) {
      observedOpen = true;
      // Playwright exposes creation, frames and close, but no handshake-open
      // event. The first frame proves communication; this is its timestamp.
      emit("websocket-open", { observed: "first-frame" });
    }
  };
  socket.on("framereceived", () => frame(true));
  socket.on("framesent", () => frame(false));
  socket.on("socketerror", () => emit("websocket-error"));
  socket.on("close", () => emit("websocket-close"));
}
