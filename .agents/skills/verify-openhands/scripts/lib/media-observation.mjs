/** Install before page scripts. This function is serialized by Playwright. */
export function installMediaObservation() {
  if (window.__ohMediaObservation) return;

  const limit = 128;
  const captures = [];
  const microphones = [];
  const peers = [];
  const elements = [];
  const tracks = new WeakMap();
  const media = new WeakMap();
  const peerEntries = new WeakMap();
  const counts = { captures: 0, tracks: 0, peers: 0, elements: 0 };
  const dropped = { captures: 0, microphones: 0, peers: 0, elements: 0 };
  const now = () => new Date().toISOString();
  const keep = (list, entry, kind) => {
    list.push(entry);
    if (list.length > limit) {
      list.shift();
      dropped[kind] += 1;
    }
    return entry;
  };
  const safeState = (value, allowed) =>
    allowed.includes(value) ? value : "unknown";
  const errorName = (error) =>
    safeState(error?.name, [
      "NotAllowedError",
      "NotFoundError",
      "NotReadableError",
      "OverconstrainedError",
      "AbortError",
      "SecurityError",
      "TypeError",
      "InvalidStateError",
      "NotSupportedError",
      "PermissionDeniedError",
    ]);
  const readTrack = (track) => {
    if (!track) return null;
    let entry = tracks.get(track);
    if (!entry) {
      entry = { id: `track-${++counts.tracks}`, ref: new WeakRef(track) };
      tracks.set(track, entry);
    }
    entry.lastState = {
      id: entry.id,
      kind: safeState(track.kind, ["audio", "video"]),
      enabled: Boolean(track.enabled),
      muted: Boolean(track.muted),
      readyState: safeState(track.readyState, ["live", "ended"]),
    };
    return entry.lastState;
  };
  const streamTracks = (stream) => {
    if (!stream || typeof stream.getTracks !== "function") return [];
    return stream.getTracks().map(readTrack);
  };
  const readElement = (element) => ({
    connected: Boolean(element.isConnected),
    paused: Boolean(element.paused),
    muted: Boolean(element.muted),
    ended: Boolean(element.ended),
    readyState: Number.isInteger(element.readyState)
      ? element.readyState
      : null,
    hasSrcObject: Boolean(element.srcObject),
    srcObjectTracks: streamTracks(element.srcObject),
  });
  const observeElement = (element) => {
    let entry = media.get(element);
    if (!entry) {
      entry = keep(
        elements,
        {
          id: `media-${++counts.elements}`,
          ref: new WeakRef(element),
        },
        "elements",
      );
      media.set(element, entry);
      for (const type of [
        "playing",
        "pause",
        "ended",
        "emptied",
        "volumechange",
      ])
        element.addEventListener(type, () => {
          const current = entry.ref.deref();
          if (current)
            safely(() => {
              entry.lastState = readElement(current);
            });
        });
    }
    entry.lastState = readElement(element);
    return entry;
  };
  const readPeer = (peer) => ({
    connectionState: safeState(peer.connectionState, [
      "new",
      "connecting",
      "connected",
      "disconnected",
      "failed",
      "closed",
    ]),
    iceConnectionState: safeState(peer.iceConnectionState, [
      "new",
      "checking",
      "connected",
      "completed",
      "disconnected",
      "failed",
      "closed",
    ]),
    iceGatheringState: safeState(peer.iceGatheringState, [
      "new",
      "gathering",
      "complete",
    ]),
    signalingState: safeState(peer.signalingState, [
      "stable",
      "have-local-offer",
      "have-remote-offer",
      "have-local-pranswer",
      "have-remote-pranswer",
      "closed",
    ]),
    senders: peer.getSenders().map((sender) => readTrack(sender.track)),
    receivers: peer.getReceivers().map((receiver) => readTrack(receiver.track)),
  });
  const observePeer = (peer) => {
    const entry = keep(
      peers,
      {
        id: `peer-${++counts.peers}`,
        ref: new WeakRef(peer),
        lastState: readPeer(peer),
      },
      "peers",
    );
    peerEntries.set(peer, entry);
    for (const type of [
      "connectionstatechange",
      "iceconnectionstatechange",
      "signalingstatechange",
    ])
      peer.addEventListener(type, () => {
        const current = entry.ref.deref();
        if (current)
          safely(() => {
            entry.lastState = readPeer(current);
          });
      });
  };
  // Observation must never replace a native return value or throw into the App.
  const safely = (callback) => {
    try {
      return callback();
    } catch {
      return undefined;
    }
  };
  const supportsWeakRefs = typeof WeakRef === "function";
  const originalPlay = HTMLMediaElement.prototype.play;
  window.__ohMediaPlays = [];
  HTMLMediaElement.prototype.play = function play(...args) {
    window.__ohMediaPlays.push({
      src: this.currentSrc || this.src || "",
      ts: now(),
    });
    const entry = supportsWeakRefs ? safely(() => observeElement(this)) : null;
    const result = Reflect.apply(originalPlay, this, args);
    if (entry && result?.then) {
      entry.playState = "pending";
      result.then(
        () => {
          entry.playState = "fulfilled";
          delete entry.playErrorName;
        },
        (error) => {
          entry.playState = "rejected";
          entry.playErrorName = errorName(error);
        },
      );
    }
    return result;
  };

  if (supportsWeakRefs) {
    // A detached Audio object can receive a stream before play() is called.
    const srcObject = Object.getOwnPropertyDescriptor(
      HTMLMediaElement.prototype,
      "srcObject",
    );
    if (srcObject?.set && srcObject.configurable) {
      Object.defineProperty(HTMLMediaElement.prototype, "srcObject", {
        ...srcObject,
        set(value) {
          Reflect.apply(srcObject.set, this, [value]);
          safely(() => observeElement(this));
        },
      });
    }
    for (const [name, observer] of [
      ["Audio", observeElement],
      ["RTCPeerConnection", observePeer],
    ]) {
      const original = window[name];
      if (typeof original !== "function") continue;
      window[name] = new Proxy(original, {
        construct(target, args, newTarget) {
          const instance = Reflect.construct(target, args, newTarget);
          safely(() => observer(instance));
          return instance;
        },
      });
    }
    const peerPrototype = window.RTCPeerConnection?.prototype;
    if (peerPrototype?.close) {
      const originalClose = peerPrototype.close;
      peerPrototype.close = function close(...args) {
        const result = Reflect.apply(originalClose, this, args);
        const entry = peerEntries.get(this);
        if (entry)
          safely(() => {
            entry.lastState = readPeer(this);
          });
        return result;
      };
    }
    // stop() need not fire an ended event. Keep the final state for a later
    // snapshot even if the stopped track is garbage-collected in between.
    const trackPrototype = window.MediaStreamTrack?.prototype;
    if (trackPrototype?.stop) {
      const originalStop = trackPrototype.stop;
      trackPrototype.stop = function stop(...args) {
        const result = Reflect.apply(originalStop, this, args);
        if (tracks.has(this)) safely(() => readTrack(this));
        return result;
      };
    }
    const devicePrototype =
      window.MediaDevices?.prototype ?? navigator.mediaDevices;
    if (devicePrototype?.getUserMedia) {
      const original = devicePrototype.getUserMedia;
      devicePrototype.getUserMedia = function getUserMedia(...args) {
        const entry = keep(
          captures,
          {
            id: `capture-${++counts.captures}`,
            requestedAt: now(),
            audioRequested: Boolean(args[0]?.audio),
            videoRequested: Boolean(args[0]?.video),
            status: "pending",
            audioTrackIds: [],
          },
          "captures",
        );
        let result;
        try {
          result = Reflect.apply(original, this, args);
        } catch (error) {
          entry.status = "error";
          entry.errorName = errorName(error);
          throw error;
        }
        result.then(
          (stream) => {
            entry.status = "granted";
            safely(() => {
              for (const track of stream.getAudioTracks()) {
                const state = readTrack(track);
                entry.audioTrackIds.push(state.id);
                keep(
                  microphones,
                  {
                    captureRequestId: entry.id,
                    track: tracks.get(track),
                  },
                  "microphones",
                );
              }
            });
          },
          (error) => {
            entry.status = "error";
            entry.errorName = errorName(error);
          },
        );
        return result;
      };
    }
  }

  const inboundStats = async (peer) => {
    if (peer.connectionState === "closed")
      return { statsStatus: "closed", inboundAudio: [] };
    let timer;
    try {
      const report = await Promise.race([
        peer.getStats(),
        new Promise((resolve) => {
          timer = setTimeout(() => resolve(null), 1500);
        }),
      ]);
      if (!report) return { statsStatus: "timeout", inboundAudio: [] };
      const inboundAudio = [];
      report.forEach((row) => {
        if (
          row.type !== "inbound-rtp" ||
          (row.kind ?? row.mediaType) !== "audio"
        )
          return;
        const item = {};
        for (const field of [
          "bytesReceived",
          "packetsReceived",
          "totalAudioEnergy",
          "totalSamplesReceived",
        ])
          if (
            typeof row[field] === "number" &&
            Number.isFinite(row[field]) &&
            row[field] >= 0
          )
            item[field] = row[field];
        inboundAudio.push(item);
      });
      return { statsStatus: "ok", inboundAudio };
    } catch {
      return { statsStatus: "error", inboundAudio: [] };
    } finally {
      clearTimeout(timer);
    }
  };
  const microphonePermission = async () => {
    if (!navigator.permissions?.query) return "unsupported";
    try {
      const status = await navigator.permissions.query({ name: "microphone" });
      return safeState(status.state, ["granted", "denied", "prompt"]);
    } catch {
      return "unsupported";
    }
  };
  window.__ohMediaObservation = {
    async snapshot() {
      if (supportsWeakRefs)
        for (const element of document.querySelectorAll("audio,video"))
          safely(() => observeElement(element));
      return {
        observedAt: now(),
        supported: supportsWeakRefs,
        secureContext: Boolean(window.isSecureContext),
        microphonePermission: await microphonePermission(),
        captureRequests: captures.map((entry) => ({
          ...entry,
          audioTrackIds: [...entry.audioTrackIds],
        })),
        microphoneTracks: microphones.map(({ captureRequestId, track }) => {
          const current = track.ref.deref();
          return {
            captureRequestId,
            collected: !current,
            ...(current ? readTrack(current) : track.lastState),
          };
        }),
        peerConnections: await Promise.all(
          peers.map(async (entry) => {
            const current = entry.ref.deref();
            if (!current)
              return {
                id: entry.id,
                collected: true,
                ...entry.lastState,
                statsStatus: "collected",
                inboundAudio: [],
              };
            entry.lastState = readPeer(current);
            return {
              id: entry.id,
              collected: false,
              ...entry.lastState,
              ...(await inboundStats(current)),
            };
          }),
        ),
        audioElements: elements.map((entry) => {
          const current = entry.ref.deref();
          if (current) entry.lastState = readElement(current);
          return {
            id: entry.id,
            collected: !current,
            ...entry.lastState,
            ...(entry.playState ? { playState: entry.playState } : {}),
            ...(entry.playErrorName
              ? { playErrorName: entry.playErrorName }
              : {}),
          };
        }),
        dropped: { ...dropped },
      };
    },
  };
}
