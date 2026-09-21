#!/bin/bash
set -euxo pipefail
# Build keplearn from the pinned public release tag (no vendored source or
# binaries in the srbench tree, per CONTRIBUTING). The crate has no
# dependencies, so the only network fetches are this clone and nothing else;
# rust/cargo come from conda-forge via environment.yml.
TAG=v0.1.0
git clone --depth 1 --branch "$TAG" https://github.com/owls-on-wires/keplearn.git keplearn-src
cd keplearn-src
cargo build --release
install -m 0755 target/release/keplearn "${CONDA_PREFIX}/bin/keplearn"
cd ..
rm -rf keplearn-src
# The toolchain is only needed for this build. Removing it here keeps the
# image layer near the base size instead of ~3.2 GB, because the environment
# install and this script run inside the same Docker RUN layer.
micromamba remove -y -n base rust compilers git
keplearn --version
echo "keplearn: installed ${TAG} from source at ${CONDA_PREFIX}/bin/keplearn"
