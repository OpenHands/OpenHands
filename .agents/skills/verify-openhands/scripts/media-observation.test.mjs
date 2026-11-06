import assert from "node:assert/strict";
import { createContext, runInContext } from "node:vm";
import { afterEach, test, vi } from "vitest";
import { installMediaObservation } from "./lib/media-observation.mjs";

// These controlled native API doubles exercise observer transparency and the
// output boundary. They are harness unit tests, never evidence of live audio.
function realm() {
  class Track {
    constructor() {
      this.kind = "audio";
      this.enabled = true;
      this.muted = false;
      this.readyState = "live";
      this.id = "private-device-track";
      this.label = "private microphone label";
    }
    stop() {
      this.readyState = "ended";
    }
  }
  class Media extends EventTarget {
    constructor() {
      super();
      this.paused = true;
      this.muted = false;
      this.readyState = 0;
      this.ended = false;
      this.isConnected = false;
      this.currentSrc = "";
      this.src = "";
      this.stream = null;
    }
    get srcObject() {
      return this.stream;
    }
    set srcObject(stream) {
      this.stream = stream;
    }
    play(...args) {
      this.playArgs = args;
      this.paused = false;
      this.playPromise = this.nextPlay ?? Promise.resolve();
      return this.playPromise;
    }
    pause() {
      this.paused = true;
    }
  }
  class Audio extends Media {}
  class Devices {
    getUserMedia(...args) {
      this.lastArgs = args;
      this.lastThis = this;
      if (this.syncError) throw this.syncError;
      return this.result;
    }
  }
  class Peer extends EventTarget {
    static sentinel = "preserved-static-value";
    constructor(...args) {
      super();
      this.constructorArgs = args;
      this.connectionState = "new";
      this.iceConnectionState = "new";
      this.iceGatheringState = "new";
      this.signalingState = "stable";
      this.senders = [];
      this.receivers = [];
      this.report = new Map();
      this.statsCalls = 0;
      this.localDescription = { sdp: "private-sdp" };
    }
    getSenders() {
      return this.senders;
    }
    getReceivers() {
      return this.receivers;
    }
    getStats() {
      this.statsCalls += 1;
      return this.pendingStats ?? Promise.resolve(this.report);
    }
    close() {
      this.connectionState = "closed";
      this.iceConnectionState = "closed";
      this.signalingState = "closed";
      this.dispatchEvent(new Event("connectionstatechange"));
    }
  }
  const devices = new Devices();
  const permissions = {
    state: "granted",
    async query(descriptor) {
      this.lastDescriptor = descriptor;
      return { state: this.state };
    },
  };
  const context = createContext({
    HTMLMediaElement: Media,
    MediaStreamTrack: Track,
    MediaDevices: Devices,
    Audio,
    RTCPeerConnection: Peer,
    navigator: { mediaDevices: devices, permissions },
    document: { querySelectorAll: () => [] },
    setTimeout,
    clearTimeout,
    isSecureContext: true,
  });
  context.window = context;
  runInContext(`(${installMediaObservation.toString()})()`, context);
  const snapshot = async () =>
    JSON.parse(JSON.stringify(await context.__ohMediaObservation.snapshot()));
  const stream = (track) => ({
    getTracks: () => [track],
    getAudioTracks: () => [track],
  });
  return {
    context,
    Track,
    Media,
    Peer,
    devices,
    permissions,
    stream,
    snapshot,
  };
}

afterEach(() => vi.useRealTimers());

test("capture preserves the native promise, receiver and arguments while tracking mute and stop", async () => {
  const { context, Track, devices, stream, snapshot } = realm();
  const track = new Track();
  const value = stream(track);
  const nativeResult = Promise.resolve(value);
  devices.result = nativeResult;
  const constraints = {
    audio: { deviceId: "private-device-id" },
    video: false,
  };
  const returned = context.navigator.mediaDevices.getUserMedia(constraints);
  assert.equal(returned, nativeResult);
  assert.equal(await returned, value);
  assert.equal(devices.lastThis, devices);
  assert.equal(devices.lastArgs[0], constraints);
  let data = await snapshot();
  assert.equal(data.captureRequests[0].status, "granted");
  assert.equal(data.captureRequests[0].audioRequested, true);
  assert.equal(data.microphoneTracks[0].readyState, "live");
  assert.equal(data.microphoneTracks[0].enabled, true);
  track.enabled = false;
  assert.equal((await snapshot()).microphoneTracks[0].enabled, false);
  track.stop();
  data = await snapshot();
  assert.equal(data.microphoneTracks[0].readyState, "ended");
  assert.doesNotMatch(JSON.stringify(data), /private-device|microphone label/);
});

test("permission rejection and synchronous native errors keep identity and expose no private detail", async () => {
  const { context, devices, permissions, snapshot } = realm();
  permissions.state = "denied";
  const error = Object.assign(new Error("private-provider-url-and-token"), {
    name: "NotAllowedError",
  });
  const nativeResult = Promise.reject(error);
  devices.result = nativeResult;
  const returned = context.navigator.mediaDevices.getUserMedia({ audio: true });
  assert.equal(returned, nativeResult);
  await assert.rejects(returned, (observed) => observed === error);
  devices.syncError = Object.assign(new Error("private-constraint"), {
    name: "private-error-name",
  });
  assert.throws(
    () => devices.getUserMedia({ audio: true }),
    (observed) => observed === devices.syncError,
  );
  const data = await snapshot();
  assert.equal(data.microphonePermission, "denied");
  assert.equal(permissions.lastDescriptor.name, "microphone");
  assert.equal(data.captureRequests[0].errorName, "NotAllowedError");
  assert.equal(data.captureRequests[1].errorName, "unknown");
  assert.equal(data.microphoneTracks.length, 0);
  assert.doesNotMatch(JSON.stringify(data), /private-/);
});

test("detached Audio and createElement-style media report real output and stream cleanup", async () => {
  const { context, Media, Track, stream, snapshot } = realm();
  const audio = new context.Audio();
  assert.ok(audio instanceof Media);
  const remote = new Track();
  audio.srcObject = stream(remote);
  const result = audio.play("native-argument");
  assert.equal(result, audio.playPromise);
  assert.deepEqual(audio.playArgs, ["native-argument"]);
  await result;
  // An element created outside the Audio constructor is observed at srcObject.
  const created = new Media();
  created.srcObject = stream(new Track());
  let data = await snapshot();
  assert.equal(data.audioElements.length, 2);
  assert.equal(data.audioElements[0].connected, false);
  assert.equal(data.audioElements[0].playState, "fulfilled");
  assert.equal(data.audioElements[0].hasSrcObject, true);
  assert.equal(data.audioElements[0].srcObjectTracks[0].readyState, "live");
  audio.muted = true;
  audio.pause();
  data = await snapshot();
  assert.equal(data.audioElements[0].paused, true);
  assert.equal(data.audioElements[0].muted, true);
  audio.srcObject = null;
  data = await snapshot();
  assert.equal(data.audioElements[0].hasSrcObject, false);
  assert.deepEqual(data.audioElements[0].srcObjectTracks, []);
  assert.equal(context.__ohMediaPlays.length, 1);
});

test("peer observation retains constructor semantics and permits only inbound audio counters", async () => {
  const { context, Peer, Track, snapshot } = realm();
  const configuration = {
    iceServers: [
      { urls: "private-ice-address", credential: "private-credential" },
    ],
  };
  const peer = new context.RTCPeerConnection(configuration);
  assert.ok(peer instanceof Peer);
  assert.equal(peer.constructorArgs[0], configuration);
  assert.equal(context.RTCPeerConnection.sentinel, Peer.sentinel);
  peer.connectionState = "connected";
  peer.iceConnectionState = "completed";
  peer.senders = [{ track: new Track() }];
  peer.receivers = [{ track: new Track() }];
  peer.report = new Map([
    [
      "private-report-id",
      {
        type: "inbound-rtp",
        kind: "audio",
        bytesReceived: 345,
        packetsReceived: 12,
        totalAudioEnergy: 0.125,
        totalSamplesReceived: 960,
        id: "private-report-id",
        trackIdentifier: "private-track-id",
        remoteAddress: "private-ip",
        arbitrary: "private-transcript",
        timestamp: 1000,
      },
    ],
    ["video", { type: "inbound-rtp", kind: "video", bytesReceived: 999 }],
    ["outbound", { type: "outbound-rtp", kind: "audio", bytesReceived: 888 }],
    ["candidate", { type: "remote-candidate", address: "private-address" }],
  ]);
  const data = await snapshot();
  const observed = data.peerConnections[0];
  assert.equal(observed.connectionState, "connected");
  assert.equal(observed.iceConnectionState, "completed");
  assert.equal(observed.senders[0].readyState, "live");
  assert.equal(observed.statsStatus, "ok");
  assert.deepEqual(observed.inboundAudio, [
    {
      bytesReceived: 345,
      packetsReceived: 12,
      totalAudioEnergy: 0.125,
      totalSamplesReceived: 960,
    },
  ]);
  assert.doesNotMatch(
    JSON.stringify(data),
    /private-|sdp|credential|timestamp|Address/,
  );
  peer.close();
  const closed = (await snapshot()).peerConnections[0];
  assert.equal(closed.connectionState, "closed");
  assert.equal(closed.statsStatus, "closed");
  assert.equal(peer.statsCalls, 1);
});

test("stats omit unsupported or nonnumeric counters and bound a stalled report", async () => {
  vi.useFakeTimers();
  const { context, snapshot } = realm();
  const peer = new context.RTCPeerConnection();
  peer.report = new Map([
    [
      "audio",
      {
        type: "inbound-rtp",
        mediaType: "audio",
        packetsReceived: 2,
        bytesReceived: "private-value",
        totalAudioEnergy: NaN,
        totalSamplesReceived: -1,
      },
    ],
  ]);
  assert.deepEqual((await snapshot()).peerConnections[0].inboundAudio, [
    { packetsReceived: 2 },
  ]);
  peer.pendingStats = new Promise(() => {});
  const pending = snapshot();
  await vi.advanceTimersByTimeAsync(1501);
  assert.equal((await pending).peerConnections[0].statsStatus, "timeout");
  peer.pendingStats = Promise.reject(new Error("private-stats-error"));
  const failed = await snapshot();
  assert.equal(failed.peerConnections[0].statsStatus, "error");
  assert.doesNotMatch(JSON.stringify(failed), /private-/);
  assert.equal(vi.getTimerCount(), 0);
});

test("installation is idempotent, history is bounded and play rejection remains native", async () => {
  const { context, devices, Track, stream, snapshot } = realm();
  const method = devices.getUserMedia;
  runInContext(`(${installMediaObservation.toString()})()`, context);
  assert.equal(devices.getUserMedia, method);
  for (let i = 0; i < 130; i += 1) {
    devices.result = Promise.resolve(stream(new Track()));
    await devices.getUserMedia({ audio: true });
  }
  const audio = new context.Audio();
  const error = Object.assign(new Error("private-playback-detail"), {
    name: "NotAllowedError",
  });
  audio.nextPlay = Promise.reject(error);
  const returned = audio.play();
  assert.equal(returned, audio.nextPlay);
  await assert.rejects(returned, (observed) => observed === error);
  const data = await snapshot();
  assert.equal(data.captureRequests.length, 128);
  assert.equal(data.microphoneTracks.length, 128);
  assert.equal(data.dropped.captures, 2);
  assert.equal(data.dropped.microphones, 2);
  assert.equal(data.audioElements[0].playState, "rejected");
  assert.equal(data.audioElements[0].playErrorName, "NotAllowedError");
  assert.doesNotMatch(JSON.stringify(data), /private-/);
});
