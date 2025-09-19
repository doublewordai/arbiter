# CPU Build Target
FROM rust:1.88 AS base-cpu
RUN cargo install sccache --version ^0.7
RUN cargo install cargo-chef --version ^0.1
ENV RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache

FROM base-cpu AS planner-cpu
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef prepare --recipe-path recipe.json

FROM base-cpu AS builder-cpu
WORKDIR /app
COPY --from=planner-cpu /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo build --release

# GPU Build Target
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base-gpu
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo install sccache --version ^0.7
RUN cargo install cargo-chef --version ^0.1
ENV RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache

FROM base-gpu AS planner-gpu
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef prepare --recipe-path recipe.json

FROM base-gpu AS builder-gpu
WORKDIR /app
COPY --from=planner-gpu /app/recipe.json recipe.json
ENV CUDA_COMPUTE_CAP="80"
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef cook --release --features cuda --recipe-path recipe.json
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo build --release --features cuda

# CPU Runtime
FROM debian:bookworm-slim AS cpu
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false appuser
WORKDIR /app
COPY --from=builder-cpu /app/target/release/arbiter /app/arbiter
RUN chown appuser:appuser /app/arbiter
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser

USER appuser
EXPOSE 3000
ENTRYPOINT ["./arbiter"]

# GPU Runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS gpu
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false appuser
WORKDIR /app
COPY --from=builder-gpu /app/target/release/arbiter /app/arbiter
RUN chown appuser:appuser /app/arbiter
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser

USER appuser
EXPOSE 3000
ENTRYPOINT ["./arbiter"]