/**
 * Request guards and JSON helpers shared by the provider bridges the Agent
 * Canvas server hosts for the web build (scripts/mars-web-bridge.mjs for
 * DigitalOcean, scripts/maritime-web-bridge.mjs for Maritime). Both hold a
 * provider credential and expose it same-origin only, so both refuse
 * cross-origin and DNS-rebound requests the same way.
 */

import { timingSafeEqual } from "node:crypto";

import { isLoopbackBind } from "./bind-host.mjs";
import { readRequestBody } from "./http-body.mjs";

export function safeEqual(a, b) {
  const x = Buffer.from(a);
  const y = Buffer.from(b);
  return x.length === y.length && timingSafeEqual(x, y);
}

/** Hostname of the request's `Host` header, without port or brackets. */
export function requestHostname(req) {
  const host = req.headers.host ?? "";
  try {
    return new URL(`http://${host}`).hostname.replace(/^\[|\]$/g, "");
  } catch {
    return "";
  }
}

/**
 * A page on another origin cannot be allowed to drive a bridge or read
 * through its proxy. Same-origin requests either carry a matching Origin or
 * none at all (plain navigations, same-origin GETs).
 */
export function isCrossOrigin(req) {
  const origin = req.headers.origin;
  if (!origin) return false;
  try {
    return new URL(origin).host !== req.headers.host;
  } catch {
    return true;
  }
}

/**
 * On a loopback bind the only legitimate `Host` values are loopback names;
 * a DNS-rebound page arrives with its own hostname in both `Host` and
 * `Origin`, so the origin check alone would pass it.
 */
export function isForeignHost(req, bindHost) {
  if (!isLoopbackBind(bindHost)) return false;
  const hostname = requestHostname(req);
  return !(isLoopbackBind(hostname) || hostname.endsWith(".localhost"));
}

export function writeJson(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(payload),
    "Cache-Control": "no-store",
  });
  res.end(payload);
}

export async function readJsonBody(req, limit) {
  const body = await readRequestBody(req, limit);
  if (body.length === 0) return {};
  try {
    return JSON.parse(body.toString("utf8"));
  } catch {
    throw new Error("request body is not valid JSON");
  }
}
