#!/usr/bin/env bash
# Downloads or updates the Ethos-U core driver under third_party/.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEST="${REPO_ROOT}/third_party/ethos-u-core-driver"
REPO_URL="${ETHOS_U_REPO:-https://github.com/ARM-software/ethos-u.git}"
REF="${ETHOS_U_REF:-main}"
ARCHIVE="${ETHOS_U_ARCHIVE:-}"

fallback_download_extract() {
    echo "Falling back to ZIP download from codeload.github.com"
    tmpzip=$(mktemp /tmp/ethos-u-XXXXXX.zip)
    # Prefer requested REF if it's a branch; otherwise default to main
    zip_url="https://codeload.github.com/ARM-software/ethos-u/zip/refs/heads/main"
    if [[ "${REF}" =~ ^(main|master)$ ]]; then
        zip_url="https://codeload.github.com/ARM-software/ethos-u/zip/refs/heads/${REF}"
    fi
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${zip_url}" -o "${tmpzip}" || {
            echo "error: curl download failed from ${zip_url}" >&2
            return 1
        }
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "${tmpzip}" "${zip_url}" || {
            echo "error: wget download failed from ${zip_url}" >&2
            return 1
        }
    else
        echo "error: neither curl nor wget found for ZIP fallback" >&2
        return 1
    fi

    tmpdir=$(mktemp -d)
    cleanup() {
        rm -rf "${tmpdir}" "${tmpzip}"
    }
    trap cleanup EXIT

    unzip -q "${tmpzip}" -d "${tmpdir}"
    mkdir -p "${REPO_ROOT}/third_party"
    rm -rf "${DEST}"
    extracted_root="$(find "${tmpdir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [ -n "${extracted_root}" ]; then
        mv "${extracted_root}" "${DEST}"
    else
        echo "error: failed to locate extracted root directory" >&2
        return 1
    fi
    echo "Ethos-U driver extracted to ${DEST}"
}

if [ -n "${ARCHIVE}" ]; then
    if [ ! -f "${ARCHIVE}" ]; then
        echo "error: ETHOS_U_ARCHIVE points to a missing file: ${ARCHIVE}" >&2
        exit 1
    fi

    tmpdir=$(mktemp -d)
    cleanup() {
        rm -rf "${tmpdir}"
    }
    trap cleanup EXIT

    echo "Extracting Ethos-U driver from ${ARCHIVE}"
    case "${ARCHIVE}" in
        *.zip)
            unzip -q "${ARCHIVE}" -d "${tmpdir}"
            ;;
        *.tar.gz|*.tgz)
            tar -xzf "${ARCHIVE}" -C "${tmpdir}"
            ;;
        *.tar.xz|*.txz)
            tar -xJf "${ARCHIVE}" -C "${tmpdir}"
            ;;
        *.tar.bz2|*.tbz2)
            tar -xjf "${ARCHIVE}" -C "${tmpdir}"
            ;;
        *)
            echo "error: unsupported archive format: ${ARCHIVE}" >&2
            exit 1
            ;;
    esac

    mkdir -p "${REPO_ROOT}/third_party"
    rm -rf "${DEST}"

    extracted_root="$(find "${tmpdir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [ -n "${extracted_root}" ] && [ "$(find "${tmpdir}" -mindepth 1 -maxdepth 1 | wc -l)" -eq 1 ]; then
        mv "${extracted_root}" "${DEST}"
    else
        mkdir -p "${DEST}"
        cp -R "${tmpdir}"/. "${DEST}/"
    fi

    echo "Ethos-U driver extracted to ${DEST}"
    exit 0
fi

if [ -d "${DEST}/.git" ]; then
    echo "Updating Ethos-U driver in ${DEST}"
    if ! git -C "${DEST}" -c http.https://github.com/.extraheader= fetch --depth 1 origin "${REF}"; then
        echo "warn: git fetch failed; attempting ZIP fallback" >&2
        fallback_download_extract || exit 1
        exit 0
    fi
    git -C "${DEST}" checkout FETCH_HEAD
else
    mkdir -p "${REPO_ROOT}/third_party"
    echo "Cloning Ethos-U driver (${REF}) into ${DEST}"
    if ! git -c http.https://github.com/.extraheader= clone --depth 1 --branch "${REF}" "${REPO_URL}" "${DEST}"; then
        echo "warn: git clone failed from ${REPO_URL}; attempting ZIP fallback" >&2
        fallback_download_extract || exit 1
        exit 0
    fi
fi

echo "Ethos-U driver is ready under ${DEST}"
