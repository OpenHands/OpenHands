#!/usr/bin/env node
// Diagnostic Cloud facade: only deployment policy and Cloud list envelopes are
// synthetic. Profiles and automation capabilities come from the private stack.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
const run = process.argv[2];
const state = JSON.parse(fs.readFileSync(path.join(run, "run.json")));
const key = fs
  .readFileSync(path.join(run, "private/session-key"), "utf8")
  .trim();
const policyFile = path.join(run, "private/profile-policy.json");
if (!fs.existsSync(policyFile))
  fs.writeFileSync(policyFile, JSON.stringify({ enabled: false }), {
    mode: 0o600,
  });
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  res.setHeader("Access-Control-Allow-Origin", state.baseUrl);
  res.setHeader(
    "Access-Control-Allow-Headers",
    req.headers["access-control-request-headers"] ||
      "authorization,content-type,x-org-id",
  );
  res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS");
  if (req.method === "OPTIONS") {
    res.writeHead(204);
    return res.end();
  }
  const policy = JSON.parse(fs.readFileSync(policyFile));
  const json = (body) => {
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify(body));
  };
  if (url.pathname === "/api/v1/web-client/config")
    return json({
      providers_configured: [],
      feature_flags: {
        deployment_mode: "self_hosted",
        enable_agent_profiles_page: policy.enabled,
      },
    });
  if (
    url.pathname === "/api/organizations" ||
    url.pathname === "/api/v1/organizations"
  )
    return json({
      items: [
        { id: "diagnostic-org", name: "Diagnostic policy control (not OHE)" },
      ],
      current_org_id: "diagnostic-org",
    });
  if (url.pathname === "/api/keys/current")
    return json({ org_id: "diagnostic-org" });
  if (url.pathname === "/api/organizations/diagnostic-org/me")
    return json({
      org_id: "diagnostic-org",
      user_id: "diagnostic",
      role: "member",
      permissions: [],
    });
  if (url.pathname === "/api/user" || url.pathname === "/api/v1/user")
    return json({
      id: "diagnostic",
      email: "diagnostic@example.invalid",
      user_consents_to_analytics: false,
    });
  if (
    url.pathname === "/api/v1/app-conversations" ||
    url.pathname.includes("app-conversations/search")
  )
    return json({ items: [], next_page_id: null });
  if (url.pathname === "/api/v1/settings")
    return json({
      is_new_user: false,
      user_consents_to_analytics: false,
      provider_tokens_set: {},
    });
  let upstream = req.url;
  if (url.pathname === "/api/organizations/diagnostic-org/profiles")
    upstream = "/api/profiles";
  if (url.pathname === "/api/v1/settings/profiles") upstream = "/api/profiles";
  try {
    const response = await fetch(state.baseUrl + upstream, {
      headers: { "X-Session-API-Key": key },
    });
    let body = await response.text();
    if (
      url.pathname.endsWith("/capabilities") &&
      policy.automationProfiles === false
    ) {
      const data = JSON.parse(body);
      data.features = data.features.filter(
        (x) => !["agentProfiles", "presetAgentProfiles"].includes(x),
      );
      body = JSON.stringify(data);
    }
    res.writeHead(response.status, {
      "content-type":
        response.headers.get("content-type") || "application/json",
    });
    res.end(body);
  } catch {
    res.writeHead(502);
    res.end("diagnostic upstream unavailable");
  }
});
server.listen(Number(process.argv[3] || 0), "127.0.0.1", () => {
  fs.writeFileSync(
    path.join(run, "private/profile-facade.json"),
    JSON.stringify({ port: server.address().port, pid: process.pid }),
    { mode: 0o600 },
  );
  console.log(
    JSON.stringify({
      diagnostic: true,
      port: server.address().port,
      pid: process.pid,
    }),
  );
});
