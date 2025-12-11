FROM python:3.10-slim

LABEL maintainer="Arhaan <arhaan.shah@gmail.com>" \
      description="JustiFlicks (dev image) - reproducible env"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG USER=app
ARG UID=1000
RUN useradd -m -u ${UID} ${USER}
WORKDIR /home/${USER}/app
USER ${USER}

COPY --chown=${USER}:${USER} pyproject.toml setup.cfg requirements.txt ./

RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=${USER}:${USER} src/ ./src/
COPY --chown=${USER}:${USER} notebooks/ ./notebooks/
COPY --chown=${USER}:${USER} scripts/ ./scripts/
COPY --chown=${USER}:${USER} demo/ ./demo/
COPY --chown=${USER}:${USER} README.md ./

RUN pip install --no-cache-dir -e .

COPY --chown=${USER}:${USER} tests/smoke_test.sh ./tests/smoke_test.sh
RUN chmod +x ./tests/smoke_test.sh

EXPOSE 8501
ENTRYPOINT ["bash", "-lc"]
CMD ["./tests/smoke_test.sh"]
