# Copyright (C) 2020 Maxime Robeyns

# A simple image for writing documentation
#
# By default, this container will simply build the HTML output and exit.
# To run it in file watcher mode (for HTML), provide the `watch` argument.
# You must bind the source and build directories, for example:
#
# Build with
#
#       docker build -f Dockerfile -t docs-writer .
#
# Run with
#
# docker run --rm -v ../bdrl:/bdrl docs/source:/docs/source -v build:/docs/build docs-writer watch

# Package installations
FROM alpine:latest
MAINTAINER Maxime Robeyns <maximerobeyns@gmail.com>
RUN apk update
RUN apk add --no-cache make python3 python3-dev py3-pip libffi-dev openssl-dev \
    inotify-tools
RUN pip3 install -U sphinx sphinx_rtd_theme live-server
# BDR requirements
RUN pip3 install -U torch gin-config

# To keep this container ephemeral, we supply the source when the container is
# mounted, and the build directory is also backed by the host's file system.
# Copy relevant files:
WORKDIR /docs
COPY Makefile /docs/Makefile
COPY make.bat /docs/make.bat
COPY docs.sh /usr/local/bin/docs.sh

# Create default directories to keep file watchers happy
RUN mkdir -p /docs/build/html /docs/source

# expose 8080
ENV PORT=8080

# run the script by default
ENTRYPOINT ["sh", "/usr/local/bin/docs.sh"]
