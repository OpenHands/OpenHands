# Develop in Docker

> [!WARNING]
> This way of running OpenHands is not officially supported. It is maintained by the community and may not work.

Install [Docker](https://docs.docker.com/engine/install/) on your host machine and run:

```bash
make docker-dev
# same as:
cd ./containers/dev
./dev.sh
```

It could take some time if you are running for the first time as Docker will pull all the  tools required for building OpenHands. The next time you run again, it should be instant.

## Build and run

If everything goes well, you should be inside a container after Docker finishes building the `openhands:dev` image similar to the following:

```bash
Build and run in Docker ...
root@93fc0005fcd2:/app#
```

You may now proceed with the normal [build and run](../../Development.md) workflow as if you were on the host.

## Make changes

The source code on the host is mounted as `/app` inside docker. You may edit the files as usual either inside the Docker container or on your host with your favorite IDE/editors.

The following are also mapped as readonly from your host:

```yaml
# host credentials
- $HOME/.git-credentials:/root/.git-credentials:ro
- $HOME/.gitconfig:/root/.gitconfig:ro
- $HOME/.npmrc:/root/.npmrc:ro
```

## VSCode

Alternatively, if you use VSCode, you could also [attach to the running container](https://code.visualstudio.com/docs/devcontainers/attach-container).

See details for [developing in docker](https://code.visualstudio.com/docs/devcontainers/containers) or simply ask `OpenHands` ;-)

## Rebuild dev image

You could optionally pass additional options to the build script.

```bash
make docker-dev OPTIONS="--build"
# or
./containers/dev/dev.sh --build
```

See [docker compose run](https://docs.docker.com/reference/cli/docker/compose/run/) for more options.
