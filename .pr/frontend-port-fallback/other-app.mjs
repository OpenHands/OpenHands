// Stand-in for an unrelated local app that already owns a port.
// Usage: node other-app.mjs <port> <host> [<host> ...]
import http from "node:http";

const [port, ...hosts] = process.argv.slice(2);
for (const host of hosts) {
  http
    .createServer((_req, res) => {
      res.setHeader("content-type", "text/html");
      res.end(
        `<body style="font:32px system-ui;padding:48px;background:#fee">` +
          `Another app listening on ${host}:${port}</body>`,
      );
    })
    .listen(Number(port), host, () =>
      console.log(`other app listening on [${host}]:${port}`),
    );
}
