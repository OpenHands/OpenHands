# Dynamically constructed Dockerfile

This folder builds a runtime image (sandbox), which will use a dynamically generated `Dockerfile`
that depends on the `base_image` **AND** a [Python source distribution](https://docs.python.org/3.10/distutils/sourcedist.html) that is based on the current commit of `openhands`.

The following command will generate a `Dockerfile` file for `nikolaik/python-nodejs:python3.12-nodejs22` (the default base image), an updated `config.sh` and the runtime source distribution files/folders into `containers/runtime`:

```bash
poetry run python3 -m openhands.runtime.utils.runtime_build \
    --base_image nikolaik/python-nodejs:python3.12-nodejs22 \
    --build_folder containers/runtime
```
