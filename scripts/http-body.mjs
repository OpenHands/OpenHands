/**
 * Request-body reading with no dependencies beyond Node itself, so Electron
 * main (via mars-oauth.mjs) can use it without loading the proxy's npm
 * packages.
 */

/**
 * Read a request body of at most `limit` bytes. Rejects, and destroys the
 * request, once it grows past the limit; rejects rather than hanging when
 * the client goes away mid-body.
 *
 * @param {import("node:http").IncomingMessage} req
 * @param {number} limit
 * @returns {Promise<Buffer>}
 */
export function readRequestBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(new Error("request body too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
    req.on("close", () => {
      if (!req.complete) reject(new Error("request aborted"));
    });
  });
}
