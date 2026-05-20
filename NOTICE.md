# Attribution & Upstream Notice

## nue Project

**nue** is a fork of [OpenHands](https://github.com/OpenHands/OpenHands), maintained by the OpenHands community.

### Fork Details

- **Upstream**: https://github.com/OpenHands/OpenHands
- **Fork base commit**: `15716df79` (upstream/main as of May 2026)
- **Repository**: https://github.com/GusCayresMindsight/nue-agentic-work
- **License**: MIT (same as upstream)

### Modifications

This fork includes the following significant modifications:

1. **Enterprise Directory Removed**
   - Removed `enterprise/` directory (507 files, 130K+ lines)
   - Removed enterprise integrations: GitHub, GitLab, Jira, Linear, Slack, Bitbucket
   - Removed: Keycloak authentication, Stripe billing, database migrations
   - Cleaned CI workflows: removed enterprise-specific jobs and matrix configurations

2. **Telemetry Gating**
   - Added `NUE_DISABLE_TELEMETRY` environment variable for PostHog opt-out
   - PostHog integration remains for backward compatibility but respects `user_consents_to_analytics` setting
   - Telemetry is enabled by default (uses OSS PostHog key)

3. **Configuration & Cleanup**
   - Deprecated `openhands/server/listen.py` and `openhands/controller/` modules remain but are scheduled for upstream removal
   - Configuration directory: `~/.nue/` (auto-migrated from `~/.openhands/`)
   - V0 codebase paths cleaned up

4. **Identity & Branding**
   - Application name: nue
   - Docker image names (future): `nue/*` pattern
   - Environment variable prefix: `NUE_*` (with `OPENHANDS_*` fallback)
   - Python package name unchanged: `openhands` (internal implementation detail)

### Upstream Sync Strategy

- **Current version**: OpenHands main branch (as of May 2026)
- **Sync frequency**: As-needed basis
- **Divergence**: Significant architectural changes (enterprise removal, telemetry gating)
  - Merge conflicts expected on future upstream syncs
  - Divergence map in `.pr/` directory during active development

### OpenHands Attribution

OpenHands is an open-source project by the OpenHands community:
- **Repository**: https://github.com/OpenHands/OpenHands
- **License**: MIT
- **Documentation**: https://docs.openhands.dev/

OpenHands is built on the work of researchers and contributors. See [OpenHands CREDITS](https://github.com/OpenHands/OpenHands/blob/main/CREDITS.md) for the full list of contributors and citations.

### Third-Party Dependencies

This project uses numerous open-source libraries. See:
- `pyproject.toml` for Python dependencies
- `frontend/package.json` for Node.js dependencies
- Full license compliance via Poetry and npm package managers

All licenses are compatible with MIT.

---

**nue is built on OpenHands — AI-driven development for everyone.**
