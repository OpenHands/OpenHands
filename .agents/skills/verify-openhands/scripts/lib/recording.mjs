// Timeline and encoding for `browser record`.
//
// The daemon polls screenshots: a headless screencast emits frames only while
// something animates, so a single repaint (a menu closing on Escape) would
// never arrive. It keeps a frame only when its bytes differ from
// the previous one, so each kept frame shows the page from its capture time
// until the next change. On stop the timeline is resampled to a constant frame
// rate and piped as JPEG into ffmpeg. An ffmpeg with libx264 (the usual system
// build) writes MP4 and can add a GIF. Playwright's bundled ffmpeg encodes only
// VP8, so with it the recording is WebM.

import { spawn, spawnSync } from "node:child_process";
import { once } from "node:events";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export const RECORD_DEFAULTS = { fps: 10, maxSeconds: 600, gifWidth: 960 };

/**
 * Take paused time out of the timeline (`browser record pause|resume`).
 * @param {{ t: number }[]} frames capture times in ms; none falls in a pause
 * @param {number} endT when capture stopped, in ms
 * @param {{ at: number, ms: number }[]} cuts each pause's start and length
 */
export function withoutPauses(frames, endT, cuts) {
  const shift = (t) =>
    t - cuts.filter((c) => c.at + c.ms <= t).reduce((sum, c) => sum + c.ms, 0);
  return {
    frames: frames.map((frame) => ({ ...frame, t: shift(frame.t) })),
    endT: shift(endT),
  };
}

/**
 * Seconds each kept frame stays on screen.
 * @param {{ t: number }[]} frames capture times in ms, ascending
 * @param {number} endT when capture stopped, in ms
 * The still before the first change is cut to `leadIn` (when the recording
 * started is arbitrary), and the last frame is held for at least `hold`.
 * Anything that animates (a running conversation's pulsing dot) is a change,
 * so long waits are cut with pause/resume rather than detected.
 */
export function frameDurations(frames, endT, { leadIn = 1, hold = 1 } = {}) {
  return frames.map((frame, i) => {
    const last = i === frames.length - 1;
    const next = last ? endT : frames[i + 1].t;
    let seconds = Math.max(0, (next - frame.t) / 1000);
    if (i === 0 && !last) seconds = Math.min(seconds, leadIn);
    if (last) seconds = Math.max(seconds, hold);
    return seconds;
  });
}

/**
 * How many times to repeat each frame at `fps`. Frame ends are rounded on the
 * running total, so the video does not drift from the real timeline, and every
 * kept frame is shown at least once.
 */
export function frameRepeats(durations, fps) {
  let total = 0;
  let shown = 0;
  return durations.map((seconds) => {
    total += seconds;
    const n = Math.max(1, Math.round(total * fps) - shown);
    shown += n;
    return n;
  });
}

/** The output size: the largest frame, with even sides (yuv420p needs them). */
export function canvasSize(frames) {
  const width = Math.max(...frames.map((f) => f.width || 0));
  const height = Math.max(...frames.map((f) => f.height || 0));
  return { width: width + (width % 2), height: height + (height % 2) };
}

/** What an ffmpeg build can write, from its `-encoders` listing. */
export function encoderSupport(encoders) {
  const has = (name) =>
    new RegExp(`^\\s*V\\S*\\s+${name}\\s`, "m").test(encoders);
  if (has("libx264")) return { format: "mp4", gif: has("gif") };
  if (has("libvpx")) return { format: "webm", gif: has("gif") };
  return null;
}

/** Where Playwright installs its ffmpeg (the browsers directory). */
export function playwrightFfmpegPaths(
  env = process.env,
  platform = process.platform,
  home = homedir(),
) {
  const exe = {
    linux: "ffmpeg-linux",
    darwin: "ffmpeg-mac",
    win32: "ffmpeg-win64.exe",
  }[platform];
  if (!exe) return [];
  const cache =
    platform === "darwin"
      ? join(home, "Library", "Caches")
      : platform === "win32"
        ? env.LOCALAPPDATA || join(home, "AppData", "Local")
        : env.XDG_CACHE_HOME || join(home, ".cache");
  const root =
    env.PLAYWRIGHT_BROWSERS_PATH && env.PLAYWRIGHT_BROWSERS_PATH !== "0"
      ? env.PLAYWRIGHT_BROWSERS_PATH
      : join(cache, "ms-playwright");
  if (!existsSync(root)) return [];
  return readdirSync(root)
    .filter((dir) => dir.startsWith("ffmpeg-"))
    .sort()
    .reverse()
    .map((dir) => join(root, dir, exe));
}

/**
 * The first usable ffmpeg: $CONTROL_OPENHANDS_FFMPEG, `ffmpeg` on PATH, then
 * Playwright's bundled build. Returns { path, format, gif } or null.
 */
export function findFfmpeg(env = process.env) {
  const candidates = [
    env.CONTROL_OPENHANDS_FFMPEG,
    "ffmpeg",
    ...playwrightFfmpegPaths(env),
  ].filter(Boolean);
  for (const path of candidates) {
    const probe = spawnSync(path, ["-hide_banner", "-encoders"], {
      encoding: "utf8",
      env,
      timeout: 10_000,
    });
    if (probe.status !== 0) continue;
    const support = encoderSupport(probe.stdout);
    if (support) return { path, ...support };
  }
  return null;
}

export function ffmpegArgs(format, { fps, width, height, out }) {
  const fit = `scale=${width}:${height}:force_original_aspect_ratio=decrease,pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2`;
  const input = [
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-f",
    "image2pipe",
    "-c:v",
    "mjpeg",
    "-framerate",
    String(fps),
    "-i",
    "pipe:0", // Playwright's build has no "-" shorthand
    "-an",
  ];
  if (format === "mp4")
    return [
      ...input,
      "-vf",
      `${fit},format=yuv420p`,
      "-c:v",
      "libx264",
      "-preset",
      "veryfast",
      "-crf",
      "23",
      "-movflags",
      "+faststart",
      out,
    ];
  // Playwright's own recorder settings for its VP8-only build.
  return [
    ...input,
    "-vf",
    fit,
    "-c:v",
    "vp8",
    "-qmin",
    "0",
    "-qmax",
    "50",
    "-crf",
    "8",
    "-deadline",
    "realtime",
    "-speed",
    "8",
    "-b:v",
    "1M",
    out,
  ];
}

/** Pipe the frames into ffmpeg at a constant rate; resolves to the frame count. */
export async function encodeRecording({
  ffmpeg,
  frames,
  durations,
  fps,
  width,
  height,
  out,
}) {
  const repeats = frameRepeats(durations, fps);
  const child = spawn(
    ffmpeg.path,
    ffmpegArgs(ffmpeg.format, { fps, width, height, out }),
    { stdio: ["pipe", "ignore", "pipe"] },
  );
  let stderr = "";
  child.stderr.on("data", (chunk) => (stderr += chunk));
  // A write after ffmpeg exits fails with EPIPE; its stderr says why.
  child.stdin.on("error", () => {});
  let closed = false;
  const done = new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("close", (code) => {
      closed = true;
      if (code === 0) resolve();
      else
        reject(
          new Error(`ffmpeg exited ${code}: ${stderr.trim().slice(-600)}`),
        );
    });
  });
  const exited = done.catch(() => {});
  for (const [i, frame] of frames.entries()) {
    const data = readFileSync(frame.file);
    for (let n = 0; n < repeats[i] && !closed; n += 1)
      if (!child.stdin.write(data))
        await Promise.race([
          once(child.stdin, "drain").catch(() => {}),
          exited,
        ]);
  }
  child.stdin.end();
  await done;
  return repeats.reduce((sum, n) => sum + n, 0);
}

/** A GIF from the MP4, with its own palette, at most `width` wide. */
export async function videoToGif({ ffmpeg, video, out, fps, width }) {
  const child = spawn(
    ffmpeg.path,
    [
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-i",
      video,
      "-vf",
      `fps=${fps},scale='min(${width},iw)':-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse`,
      out,
    ],
    { stdio: ["ignore", "ignore", "pipe"] },
  );
  let stderr = "";
  child.stderr.on("data", (chunk) => (stderr += chunk));
  const [code] = await once(child, "close");
  if (code !== 0)
    throw new Error(
      `ffmpeg (gif) exited ${code}: ${stderr.trim().slice(-600)}`,
    );
}
