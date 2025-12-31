# OpenHands Directory Scan Report
Generated: 2025-12-30

## Overview
- **Total Size**: 296M
- **Total Files**: 2,892
- **Total Directories**: 734
- **Python Files**: 1,169
- **TypeScript/JavaScript Files**: 895
- **Test Files**: 370

## Directory Structure

### Root Level Directories
```
OpenHands/
├── .git/                    (262M - Git repository)
├── openhands/               (5.0M - Main Python package)
├── frontend/                (9.2M - Frontend application)
├── evaluation/              (7.7M - Evaluation benchmarks)
├── enterprise/              (5.0M - Enterprise features)
├── tests/                   (3.6M - Test suite)
├── openhands-cli/           (872K - CLI tool)
├── openhands-ui/            (564K - UI components)
├── workspace/               (188K - Workspace files)
├── third_party/             (152K - Third-party code)
├── .github/                 (116K - GitHub workflows)
├── skills/                  (96K - Skills definitions)
├── containers/              (76K - Docker containers)
├── scripts/                 (76K - Utility scripts)
├── code-server-data/        (52K - Code server data)
├── .openhands/              (48K - OpenHands config)
├── kind/                    (40K - Kubernetes configs)
├── monitoring/              (40K - Monitoring setup)
└── dev_config/              (20K - Dev configuration)
```

## Key Components

### Main Application (`openhands/`)
Core Python package with the following modules:
- `agenthub/` - Agent hub
- `app_server/` - Application server
- `controller/` - Controller logic
- `core/` - Core functionality
- `critic/` - Critic module
- `events/` - Event system
- `integrations/` - Integrations
- `llm/` - LLM integration
- `mcp/` - MCP protocol
- `memory/` - Memory management
- `microagent/` - Microagent system
- `resolver/` - Resolver
- `runtime/` - Runtime environment
- `security/` - Security features
- `server/` - Server components
- `storage/` - Storage layer
- `utils/` - Utilities

### Configuration Files
- `docker-compose.yml` - Docker Compose configuration
- `config.toml` - Main configuration
- `config.template.toml` - Configuration template
- `.env.example` - Environment variables example
- `poetry.lock` - Poetry lock file
- `pyproject.toml` - Python project configuration

### Scripts (`scripts/`)
- `backup-openhands.sh` - Backup script
- `health-check.py` - Health check script
- `monitor-gpu.sh` - GPU monitoring
- `cleanup-old-containers.sh` - Container cleanup
- `start-openhands.sh` - Start script
- `stop-openhands.sh` - Stop script
- `restart-openhands.sh` - Restart script
- `monitor-openhands.sh` - Monitoring script
- `server-monitor.sh` - Server monitoring
- `show-server-status.sh` - Status display
- `start-sglang.sh` - SGLang start script
- `test-setup.sh` - Test setup script

### Monitoring (`monitoring/`)
- `prometheus.yml` - Prometheus configuration
- `grafana/provisioning/datasources/prometheus.yml` - Grafana data source
- `grafana/provisioning/dashboards/dashboards.yml` - Dashboard configuration

### Documentation (`docs/`)
- `MONITORING.md` - Monitoring guide
- `BACKUP.md` - Backup procedures
- `TROUBLESHOOTING.md` - Troubleshooting guide

### Docker Containers (`containers/`)
- `app/Dockerfile` - Main app container
- `dev/Dockerfile` - Development container
- `agent-server/Dockerfile` - Agent server container
- `runtime/` - Runtime containers

### Evaluation Benchmarks (`evaluation/`)
Multiple benchmark suites:
- `swe_bench/` - SWE Bench
- `multi_swe_bench/` - Multi SWE Bench
- `gpqa/` - GPQA
- `humanevalfix/` - HumanEvalFix
- `commit0/` - Commit0
- `browsing_delegation/` - Browsing delegation
- `miniwob/` - MiniWoB
- `gorilla/` - Gorilla
- `biocoder/` - BioCoder
- `the_agent_company/` - The Agent Company
- `visualwebarena/` - Visual Web Arena
- `logic_reasoning/` - Logic reasoning
- `swefficiency/` - SWEfficiency
- `discoverybench/` - Discovery Bench
- `algotune/` - AlgoTune
- `nocode_bench/` - NoCode Bench
- `toolqa/` - ToolQA
- `lca_ci_build_repair/` - LCA CI Build Repair
- `bird/` - BIRD
- `visual_swe_bench/` - Visual SWE Bench
- `swe_perf/` - SWE Performance
- `scienceagentbench/` - Science Agent Bench
- `ml_bench/` - ML Bench
- `agent_bench/` - Agent Bench
- `webarena/` - Web Arena
- `mint/` - MINT
- `EDA/` - EDA
- `testgeneval/` - TestGenEval
- `gaia/` - GAIA
- `aider_bench/` - Aider Bench

## File Types

### Python Files
- **Total**: 1,169 files
- Main locations: `openhands/`, `evaluation/`, `tests/`, `enterprise/`

### Configuration Files
- YAML/YML: Docker Compose, GitHub workflows, Kubernetes
- TOML: Configuration files, pyproject.toml
- JSON: Package files, configuration
- ENV: Environment variables

### Documentation
- Markdown files: README, guides, documentation
- Multiple README files in subdirectories

### Scripts
- Shell scripts (.sh): Setup, monitoring, testing
- Python scripts: Health checks, utilities

### Lock Files
- `poetry.lock` - Python dependencies
- `bun.lock` - Bun dependencies
- `uv.lock` - UV dependencies

### Log Files
- `logs/startup.log` - Startup logs
- `code-server-data/logs/` - Code server logs

## Large Files
- `.git/objects/pack/pack-*.pack` - Git pack files (largest files)

## Git Repositories
- Main repository: `/home/noya/OpenHands/.git`
- Workspace project: `/home/noya/OpenHands/workspace/project/.git`

## Notes
- The `.git` directory is the largest component (262M)
- Main application code is in `openhands/` (5.0M)
- Frontend code is in `frontend/` (9.2M)
- Extensive evaluation suite with multiple benchmarks
- Monitoring and backup infrastructure recently added
- Multiple Docker containers for different services

