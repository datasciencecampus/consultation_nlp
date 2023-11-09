FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye
# Install the xz-utils package
RUN apt-get update && apt-get install -y xz-utils

COPY requirements.txt /tmp/pip-tmp/
RUN python -m pip --no-cache-dir install -r /tmp/pip-tmp/requirements.txt