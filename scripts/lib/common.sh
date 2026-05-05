#!/usr/bin/env bash
# shellcheck shell=bash

common_err() {
  echo "$*" >&2
  return 0
}

common_warn() {
  common_err "$@"
  return 0
}

common_info() {
  echo "$*"
  return 0
}

common_die() {
  common_err "$@"
  exit 1
}

common_require_git_repo() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 0
  fi
  common_die "This hook must run inside a Git repository."
}

common_is_allowlisted_path() {
  local path="$1"
  local ext="${path##*.}"
  ext="${ext,,}"

  case "$path" in
    testdata/*|*/testdata/*|examples/*|*/examples/*)
      return 0
      ;;
    *) ;;
  esac

  case "$ext" in
    png|jpg|jpeg|svg|webp|ico|woff|woff2|ttf)
      return 0
      ;;
    *) ;;
  esac

  return 1
}

common_require_file() {
  local file_path="${1:-}"
  local message="${2:-File is required.}"

  if [[ -n "$file_path" && -f "$file_path" ]]; then
    return 0
  fi

  common_die "$message"
}
