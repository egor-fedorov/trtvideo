#!/usr/bin/env bash
# Registry writes are confined to the two release packages passed by the workflow.
set -euo pipefail
export LC_ALL=C

die() { echo "ERROR: $*" >&2; exit 1; }

resolve() {
    local ref=$1 output digest
    if output=$(docker buildx imagetools inspect "$ref" 2>&1); then
        # The top-level digest includes the SBOM/provenance manifests, unlike a local image ID.
        digest=$(awk '/^Digest:[[:space:]]/ {print $2; exit}' <<<"$output")
        [[ $digest =~ ^sha256:[0-9a-f]{64}$ ]] || die "No registry digest for $ref: $output"
        echo "$digest"
    elif [[ $output != "ERROR: $ref: not found" ]]; then
        die "Cannot resolve $ref: $output"
    fi
}

check_source() {
    [[ $1 =~ ^[^@]+@sha256:[0-9a-f]{64}$ ]] || die "Expected repository@sha256:digest: $1"
    local actual
    actual=$(resolve "$1")
    [[ $actual == "${1##*@}" ]] || die "Source digest missing or changed: $1"
}

check_tag() {
    local source=$1 ref="${1%@*}:$2" existing
    existing=$(resolve "$ref")
    [[ -z $existing || $existing == "${source##*@}" ]] || die "Conflicting immutable tag: $ref -> $existing"
}

tag_digest() {
    local source=$1 tag=$2 ref="${1%@*}:$2" existing actual
    existing=$(resolve "$ref")
    if [[ $existing == "${source##*@}" ]]; then
        echo "Already verified: $ref -> $existing"
        return
    fi
    [[ -z $existing || $tag == latest ]] || die "Refusing to replace $ref -> $existing"
    docker buildx imagetools create --prefer-index=false -t "$ref" "$source"
    actual=$(resolve "$ref")
    [[ $actual == "${source##*@}" ]] || die "Read-back mismatch: $ref -> $actual, expected $source"
}

publish() {
    local production=$1 tools=$2 version=$3 revision=$4 prerelease=$5 source tag latest
    local tags=("$version" "v$version" "sha-$revision")
    [[ $version =~ ^[0-9][0-9A-Za-z._-]*$ && $revision =~ ^[0-9a-f]{40}$ ]] || die "Invalid release identity"
    [[ $prerelease == true || $prerelease == false ]] || die "Unknown prerelease status: $prerelease"
    [[ ${production%@*} != "${tools%@*}" ]] || die "Release packages must be distinct"
    for source in "$production" "$tools"; do
        check_source "$source"
        for tag in "${tags[@]}"; do check_tag "$source" "$tag"; done
    done

    # One barrier for both packages. In particular, do not tag production if model-tools fails.
    make test-release-gpu IMAGE="$production" MODEL_TOOLS_IMAGE="$tools"
    for source in "$production" "$tools"; do
        for tag in "${tags[@]}"; do tag_digest "$source" "$tag"; done
    done
    if [[ $prerelease == false ]]; then
        latest=$(gh api "repos/${GITHUB_REPOSITORY:?}/releases/latest" --jq .tag_name)
        [[ -n $latest ]] || die "GitHub did not return the current stable release"
        if [[ $latest == "v$version" ]]; then
            for source in "$production" "$tools"; do tag_digest "$source" latest; done
        else
            echo "Leaving latest unchanged: GitHub's current release is $latest."
        fi
    fi
}

case ${1:-} in
    resolve) [[ $# == 2 ]] || die "resolve TAG"; resolve "$2" ;;
    candidate)
        [[ $# == 3 && $3 =~ ^candidate-[0-9]+$ ]] || die "candidate IMAGE@DIGEST candidate-RUN_ID"
        check_source "$2"
        tag_digest "$2" "$3"
        ;;
    publish) [[ $# == 6 ]] || die "publish PROD@DIGEST TOOLS@DIGEST VERSION REVISION PRERELEASE"; publish "${@:2}" ;;
    *) die "Expected resolve, candidate or publish" ;;
esac
