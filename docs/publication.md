# Publication runbook

The steps that take this repository from private development to a public portfolio
repository. Everything that can be automated or prepared is already in the repository; the
steps below are the ones that need the owner, because they change what is public, rewrite
history, or change account settings. Record each completed step, with its date and
evidence, in the publication record of [release-gate.md](release-gate.md).

## Why a new repository

Release-gate item A8 found three things that a visibility change of this repository would
expose and that cannot be removed from it:

1. Early versions of four P02 fixtures, reachable from `main`, paraphrased course quiz
   questions.
2. Several pull-request descriptions name course identifiers and local tooling. GitHub keeps
   pull-request refs and the edit history of descriptions.
3. Every authored commit carries the author's institutional e-mail address.

The recommended route therefore publishes a **new repository** built from a rewritten copy of
`main`. Commit order, dates, messages and the final tree are preserved; the fixture versions,
an old blocklist entry, the e-mail address and the em dashes in commit messages are replaced.
The old pull requests stay in the private repository, which is archived, not deleted.

[`scripts/prepare_public_history.py`](../scripts/prepare_public_history.py) builds and
verifies the rewritten copy. A dry run on 2026-09-16 rewrote all 23 commits of `main`, reported
`verified: publishable`, kept the final tree identical, and gitleaks found no leaks.

## Before you start

- Every pull request is merged or closed, and CI on `main` is green.
- `uv sync` has installed the `dev` group (it provides `git-filter-repo`).
- In GitHub **Settings > Emails**, turn on *Keep my email addresses private* and *Block
  command line pushes that expose my email*. Note your no-reply address,
  `ID+USERNAME@users.noreply.github.com`.
- Use that address for all future commits: `git config --global user.email "NOREPLY"`.

In the commands below, `OWNER` is the GitHub account, `OLD_EMAIL` the address to remove and
`NOREPLY` the no-reply address.

## 1. Archive the private repository under a new name

```bash
gh repo rename genai-practice-projects-private --repo OWNER/genai-practice-projects --yes
```

The repository stays private. The name `genai-practice-projects` becomes free for the public
repository, so README links, badges and `CITATION.cff` stay correct.

## 2. Build and verify the rewritten history

```bash
uv run python -m scripts.prepare_public_history https://github.com/OWNER/genai-practice-projects-private.git ../genai-practice-projects-public.git --old-email OLD_EMAIL --new-email NOREPLY
```

Continue only if the last line is `verified: publishable`. The script creates a bare
repository and pushes nothing. `filter-repo/commit-map` inside it maps every old commit id to
its new one.

## 3. Create the new repository, still private, and push

```bash
gh repo create OWNER/genai-practice-projects --private --description "Twelve independent, tested Python projects: data preparation, classical ML, web APIs, transformers and RAG."
git -C ../genai-practice-projects-public.git push https://github.com/OWNER/genai-practice-projects.git main
```

Your existing local clone still points at the private repository. Clone the new one afresh
(`gh repo clone OWNER/genai-practice-projects`) and work there from now on.

## 4. Apply the repository settings

The same settings the private repository had, as code:

```bash
gh api -X PATCH repos/OWNER/genai-practice-projects -F allow_squash_merge=true -F allow_merge_commit=false -F allow_rebase_merge=false -F delete_branch_on_merge=true -F has_wiki=false -F has_projects=false -f squash_merge_commit_title=COMMIT_OR_PR_TITLE -f squash_merge_commit_message=COMMIT_MESSAGES
gh api -X PUT repos/OWNER/genai-practice-projects/branches/main/protection --input .github/branch-protection.json
gh api -X PUT repos/OWNER/genai-practice-projects/actions/permissions/workflow -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false
gh api -X PUT repos/OWNER/genai-practice-projects/vulnerability-alerts
gh api -X PUT repos/OWNER/genai-practice-projects/automated-security-fixes
gh repo edit OWNER/genai-practice-projects --add-topic generative-ai,machine-learning,rag,langchain,huggingface,python,portfolio
```

Then run `heavy.yml` once by hand (Actions tab, *Run workflow*) so the new repository has its
own heavy-run evidence.

## 5. Update the evidence to the new repository

The release gate cites commit ids and workflow runs of the private repository, which a public
reader cannot open. In a pull request on the new repository:

- Replace the cited commit ids with their new ids from `filter-repo/commit-map`.
- Replace the cited workflow runs with the new repository's `ci.yml` and `heavy.yml` runs.
- Tick A8, citing the script's `verified: publishable` output and the gitleaks job.
- Complete the owner's checks C8 (read the README as a stranger) and C9 (the P12 corpus is
  invented text).

## 6. Make it public

```bash
gh repo edit OWNER/genai-practice-projects --visibility public --accept-visibility-change-consequences
```

Immediately afterwards, enable the protections that exist only for public repositories:

```bash
gh api -X PUT repos/OWNER/genai-practice-projects/private-vulnerability-reporting
gh api -X PATCH repos/OWNER/genai-practice-projects --input - <<'EOF'
{"security_and_analysis": {"secret_scanning": {"status": "enabled"}, "secret_scanning_push_protection": {"status": "enabled"}}}
EOF
gh api -X PUT repos/OWNER/genai-practice-projects/actions/permissions/fork-pr-contributor-approval -f approval_policy=all_external_contributors
```

## 7. Verify, tag and release

```bash
uv run python -m scripts.release_check          # with network: A7 must report pass
git tag -a v1.1.0 -m "1.1.0" && git push origin v1.1.0
gh release create v1.1.0 --title "1.1.0" --notes-file NOTES.md
```

`NOTES.md` is the 1.1.0 section of `CHANGELOG.md`. Record A7, C10 and the release in the
publication record, then archive the private repository:

```bash
gh repo archive OWNER/genai-practice-projects-private --yes
```

## Alternative: change the visibility of this repository

Possible, but it publishes the three A8 findings permanently. If you choose it, record in A8
why each finding is accepted, then follow steps 4 (settings already applied), 6 and 7.
