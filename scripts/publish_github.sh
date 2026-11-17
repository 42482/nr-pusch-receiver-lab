#!/usr/bin/env bash
# Publish reviewed commits using the account authorized in the browser.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
task_repo=nr-pusch-receiver-lab
export GH_CONFIG_DIR="$task_root/.cache/gh"
export XDG_CACHE_HOME="$task_root/.cache/xdg" TMPDIR="$task_root/.cache/tmp"
mkdir -p "$GH_CONFIG_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
if [[ -n "$(git status --porcelain)" ]]; then
    printf 'Working tree has uncommitted files; review and commit them before publishing.\n' >&2
    exit 2
fi
task_gh="$(command -v gh || true)"
if [[ -z "$task_gh" ]]; then
    task_gh="$task_root/.cache/tools/gh"
    [[ -x "$task_gh" ]] || bash "$task_root/scripts/setup_github_cli.sh"
fi
if ! "$task_gh" auth status --hostname github.com >/dev/null 2>&1; then
    "$task_gh" auth login --hostname github.com --git-protocol https --web
fi
task_owner="$("$task_gh" api user --jq .login)"
task_url="https://github.com/$task_owner/$task_repo.git"
printf 'Publishing reviewed commits to %s/%s\n' "$task_owner" "$task_repo"
if git remote get-url origin >/dev/null 2>&1; then
    [[ "$(git remote get-url origin)" == "$task_url" ]] || {
        printf 'Existing origin differs from the selected account/repository; refusing to replace it.\n' >&2
        exit 2
    }
elif "$task_gh" repo view "$task_owner/$task_repo" >/dev/null 2>&1; then
    git remote add origin "$task_url"
else
    "$task_gh" repo create "$task_owner/$task_repo" --private --source "$task_root" --remote origin \
        --description 'Reproducible NR PUSCH receiver experiments with traditional, Transformer and grouped neural decoders'
fi
printf -v task_helper '!%q auth git-credential' "$task_gh"
git -c credential.helper= -c "credential.https://github.com.helper=$task_helper" push -u origin main
git -c credential.helper= -c "credential.https://github.com.helper=$task_helper" push origin frozen-20261116 v0.1.1
printf '\nRepository: https://github.com/%s/%s\n' "$task_owner" "$task_repo"
printf 'Private repository: add collaborators in GitHub Settings > Collaborators.\n'
