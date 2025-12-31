# syntax=docker/dockerfile:1

###
FROM ubuntu:22.04 AS dind

# https://docs.docker.com/engine/install/ubuntu/
RUN apt-get update && apt-get install -y \
	ca-certificates \
	curl \
	&& install -m 0755 -d /etc/apt/keyrings \
	&& curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
	&& chmod a+r /etc/apt/keyrings/docker.asc \
	&& echo \
		"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
		$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update && apt-get install -y \
	docker-ce \
	docker-ce-cli \
	containerd.io \
	docker-buildx-plugin \
	docker-compose-plugin \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean \
	&& apt-get autoremove -y

###
FROM dind AS openhands

ENV DEBIAN_FRONTEND=noninteractive

#
RUN apt-get update && apt-get install -y \
	bash \
    build-essential \
    curl \
	git \
	git-lfs \
	software-properties-common \
	make \
    netcat \
    sudo \
	wget \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean \
	&& apt-get autoremove -y

# https://github.com/cli/cli/blob/trunk/docs/install_linux.md
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
	&& chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& apt-get update && apt-get -y install \
    gh \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean \
  && apt-get autoremove -y

# Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip \
    && ln -s /usr/bin/python3.12 /usr/bin/python

# NodeJS >= 22.x
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs

# Poetry >= 1.8
RUN curl -fsSL https://install.python-poetry.org | python3.12 - \
    && ln -s ~/.local/bin/poetry /usr/local/bin/poetry

#
RUN <<EOF
#!/bin/bash
printf "#!/bin/bash
set +x
uname -a
docker --version
gh --version | head -n 1
git --version
#
python --version
echo node `node --version`
echo npm `npm --version`
poetry --version
netcat -h 2>&1 | head -n 1
" > /version.sh
chmod a+x /version.sh
EOF

###
FROM openhands AS dev

RUN apt-get update && apt-get install -y \
	dnsutils \
	file \
	iproute2 \
	jq \
	lsof \
	ripgrep \
	silversearcher-ag \
	vim \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean \
	&& apt-get autoremove -y

# mark /app as safe git directory to avoid pre-commit errors
RUN git config --system --add safe.directory /app

WORKDIR /app

# cache build dependencies
RUN \
  --mount=type=bind,source=./,target=/app/,rw \
  <<EOF
#!/bin/bash
make -s clean
make -s check-dependencies
make -s install-python-dependencies

# NOTE
# node_modules are .dockerignore-d therefore not mountable
# make -s install-frontend-dependencies
EOF

#
CMD ["bash"]
