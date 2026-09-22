# Release gate - private → public

The checklist the repository has to clear before it is made public. A box is ticked only where
the evidence is recorded next to it; an item that cannot be fully checked yet stays unticked with
the reason stated.

## A. Mechanical (`scripts/release_check.py`)

`ci.yml` runs `uv run python -m scripts.release_check --offline` in the Ubuntu job on every push
and pull request; the job fails if any check reports `fail`. The link check (A7) makes network
requests, so it runs only without `--offline`, by hand.

| # | Item | Status | Evidence |
|---|---|---|---|
| A1 | Clean clone → `uv sync` → `pytest -m core` green on Ubuntu and Windows; `heavy.yml` green on Ubuntu and macOS, run URLs recorded. | ☑ | **Core, clean clone:** `ci.yml` run [35311363210](https://github.com/tudorandrian/genai-practice-projects/actions/runs/35311363210) on `148328c`, the head of pull request #3 before this evidence update (the only later change is this document): `pytest -m "core and not network"` **352 passed, 31 deselected on both `ubuntu-latest` and `windows-latest`**; the same job's offline `release_check` step passed. **Heavy:** `heavy.yml` run [35311307417](https://github.com/tudorandrian/genai-practice-projects/actions/runs/35311307417) on `626b3bb`, **all three jobs green**. `mypy` is clean on both runners ("no issues found in 75 source files"). Ubuntu: 27 passed (`models or rag or network`); `demo --all` 11 ok + 1 skipped (P10: the eSpeak driver writes no audio on that runner, so P10 reports `skipped` with that reason). macOS: 25 passed, 2 xfailed; `demo --all` 11 ok + 1 skipped: P10's speech engine did not finish writing the test recording within 60 s on that runner, so P10 reported `skipped`. In run [35310825829](https://github.com/tudorandrian/genai-practice-projects/actions/runs/35310825829) on `0267ba7` (the same code; only the Ollama pin differs) macOS gave **12/12 ok**, with P10 transcribing all 73 words. `llm` job: 4 passed against Qwen2.5 1.5B served by Ollama 0.34.2. **The 2 macOS xfails are a real limitation, not a pass:** `test_index_and_ask_end_to_end` and `test_trap_question_refuses_via_relevance_gate` in `projects/p12_study_hub/tests/test_study_hub.py` are marked `xfail(sys.platform == "darwin", strict=False)` with their assertions unchanged. On macOS an on-topic question produced a flat score field (`0.4138…0.3643`, correct lesson ranked second), so P12's margin-based relevance gate cannot separate on-topic from off-topic questions there. `projects/p12_study_hub/README.md` "Limits" gives the numbers and how to re-measure. The macOS heavy job now runs `demo --all --strict` with `P10_TTS_TIMEOUT_SECONDS=180`, so a P10 skip there fails the job instead of being recorded as above. |
| A2 | `gitleaks detect` over the full history: zero findings. | ☑ | `ci.yml`'s `gitleaks` job, run [35311363210](https://github.com/tudorandrian/genai-practice-projects/actions/runs/35311363210): **"28 commits scanned. no leaks found"** (the full history of this repository plus pull request #3). Before the first push, gitleaks v8.28.0 over the rewritten history: "26 commits scanned. no leaks found". The pre-commit `gitleaks` hook also passes locally. `.env.example` is scanned (only `uv.lock` is path-allowlisted in `.gitleaks.toml`). |
| A3 | No tracked file > 100 KB; no `.ipynb`/`.pkl`/`.joblib`/`.sqlite3`/`.bin`/`.pt`/`.pth` tracked. | ☑ | `release_check` → `PASS no-large-files`, `PASS no-binary-artifacts` (2026-09-16). Enforced on every pull request by the core tests `shared/tests/test_scaffold.py::test_no_tracked_file_over_100kb` and `::test_no_tracked_binary_artifact`, and by the offline `release_check` step in `ci.yml`. The root `uv.lock` is exempt from the size rule (a generated lockfile), in the test, the check and the `check-added-large-files` hook alike. |
| A4 | Blocklist scan over every tracked file; only the two allowed hits. | ☑ | `uv run python -m shared.blocklist --all` → exit 0, no output (2026-09-16); `release_check` → `PASS blocklist`. The only files where the certificate provider's names legitimately appear, the root `README.md` and `docs/certificate.md`, are the two entries of `shared.blocklist.ALLOWED`. |
| A5 | Every dataset in every README has a source and a licence; every `metrics.txt` is newer than its project's last code change. | ☑ | `release_check` → `PASS datasets-have-licences`. That check is a smoke test (the section exists and names a licence or provenance, in a table or in prose); per-dataset completeness is covered by B and C8. `metrics-fresh` fails only when a proof is missing; a proof older than its project's last code commit is reported as `skip`, because a code change can leave output byte-identical and git then records no newer commit for the proof. The substantive evidence is the A6 run: `uv run demo --all` followed by an empty `git status --porcelain -- projects/*/output`, so every committed proof is exactly what the code produces. Tests: `shared/tests/test_release_check.py::test_metrics_fresh_fails_only_when_the_proof_is_missing`, `::test_metrics_fresh_reports_a_stale_proof_as_skip_with_the_action_to_take`, `::test_dataset_licence_check_does_not_match_mit_inside_an_ordinary_word`. |
| A6 | `uv run demo --all` completes; `output/demo-summary.md` shows 12/12 with the date. | ☑ | Windows, rag group, in a fresh clone of this repository: **12/12 ok**, `output/demo-summary.md` dated 2026-09-18, then `git status --porcelain -- projects/*/output` empty. macOS (`heavy.yml` run [35310825829](https://github.com/tudorandrian/genai-practice-projects/actions/runs/35310825829) on `0267ba7`): **12/12 ok**, with the shared metrics identical to Windows and Ubuntu (`tips_r2=0.481`, `co2_r2=0.903`, `silhouette=0.664`, `f1=0.850`). Ubuntu (same run): **11 ok + 1 skipped** (P10, eSpeak; see A1). `demo --all` exits 0 on all three, since `skipped` is not `failed`. `output/` is gitignored, so the summary file itself is local evidence; `heavy.yml` uploads it as an artifact. |
| A8 | Nothing unpublishable outside the working tree: git history, commit identities and pull-request descriptions. | ☑ | **Resolved by publishing a new repository from a rewritten history (2026-09-18).** The pre-publication security and privacy review of 2026-09-16 found three things in the private development repository that a visibility change would have published: (1) early versions of four P02 fixtures (e06, e08, e09, e15) whose questions paraphrased course quiz questions; (2) pull-request descriptions naming course identifiers, the course data mirror and local tooling; (3) the author's institutional e-mail address on every authored commit. This repository was built from that repository's `main` by `scripts/prepare_public_history.py`: 26 commits rewritten; the four fixtures have their invented versions from their first commit (`c83f7a1`); every authored commit uses the GitHub no-reply address; no commit message contains an em dash; the final tree is identical (`fa90ef5`). The script ended `verified: publishable`, and gitleaks v8.28.0 reported "26 commits scanned. no leaks found" before the first push. The private pull requests were not carried over: that repository stays private and is archived. Commit subjects keep the private pull-request numbers, such as `(#17)`; in this repository those numbers refer to different pull requests. |
| A7 | Every link in the root README and the twelve project READMEs resolves for a logged-out client. | ☑ | 2026-09-18, after the visibility change: `uv run python -m scripts.release_check` with network access reported `PASS links-resolve`, so every link in the root and project READMEs resolved for a logged-out client, including the repository's own URLs (the repository, its Actions pages and CI badge, commit history, pull-request list). Anonymous requests for the repository page and the raw README returned 200. Loopback and RFC 2606 example hosts are not links and are skipped. Tests: `test_links_check_still_fails_a_genuinely_broken_link`, `test_links_check_reports_private_repo_404_as_skip_not_fail`, `test_links_check_skips_localhost_and_example_hosts`. |

## B. Blind review (independent review agent)

| Item | Status | Evidence |
|---|---|---|
| Receives only the root README, one project README per level (P01, P06, P07, P11) and the demo summary; answers what the author claims, whether the evidence supports it, what a senior reviewer would challenge; findings become issues, fixed or accepted in an ADR. | ☑ | Run by a separate review agent that had not seen the work, given only the materials listed, against the pushed branch's rendered pages. Verdict: "close to ready". **No issues were opened, because all seven findings were fixed in-branch before merge:** (1) the CI/history claim had nothing checkable on the page, so a CI badge and links to the commit history and pull-request list were added; (2) example-output timing lines did not match the demo summary (machine-dependent), so they were removed from all twelve READMEs; (3) P11's design notes described the author's own finding as "caught in review", so they now describe the symptom; (4) Quick Start's plain `uv sync` did not cover `demo --all`, so it now names the group each tier needs; (5) model downloads had no size, so measured sizes were added; (6) Quick Start sat below the project table and was moved above it; (7) `drug_acc=1.000` had no explanation, so P04's README now explains the synthetic data's exact separability and names the test that asserts it. All seven are in `adc6e1f`. |

## C. Owner checks

| # | Item | Status | Evidence |
|---|---|---|---|
| C8 | Read the root README as a stranger: first screen says what this is, that it is practice work, and how to run it. | ☑ | 2026-09-18, at the owner's request, by a separate review agent that had not seen the work and was given only `README.md`. Verdict: pass. The first screen says what the repository is ("Twelve independent Python projects…"), that it is practice work ("learning projects built to an engineering standard, not products"), and where the run instructions are (Getting started, and each project's own README). Its one substantive finding, that the P12 table row did not mention the reuse of P02's parser, is fixed in pull request #3. |
| C9 | Open the P12 corpus and confirm every lesson is invented text. | ☑ | 2026-09-18, at the owner's request. A sample of the corpus was read: every course is set in an invented organisation (the bike-share operator Wheel & Way, Fenwick Labs and its assistant Pebble). Measured over all 33 corpus files (21,151 distinct word 8-grams) against 251,086 text files of the author's course materials: 9 shared 8-grams (0.04%). Four are the question template's header; five are generic technical phrases, such as "fit the scaler only on the training set". No passage is shared. |
| C10 | Repository settings are complete: description, topics, MIT licence detected, private vulnerability reporting, secret scanning and push protection. | ☑ | 2026-09-18, verified through the GitHub API after the visibility change: description set; seven topics; licence detected as `MIT`; private vulnerability reporting enabled; secret scanning and push protection enabled (no alerts); workflows from outside contributors' pull requests need approval (`all_external_contributors`); branch protection as in `.github/branch-protection.json`. |

## Summary

Every item is ticked: A1-A8, B and C8-C10, each with its evidence recorded next to it.

**Re-checking later:** `uv run python -m scripts.release_check` with network access; A7
holds only while `links-resolve` reports `pass`.

## History

- **2026-09-16, gate checks.** `release_check` first failed A5 and A7 on over-specified rules.
  The licence check required a table row, so prose now counts too, and the short licence names
  got `\b` word boundaries so "MIT" no longer matches inside "committed". The link check had
  treated example hosts and the private repository's own URLs as broken. `metrics-fresh` could
  fail with no honest fix (see A5), so a stale proof is now `skip` and only a missing proof fails.
  Each check still fails on a real defect, and a regression test pins each change.
- **2026-09-16, heavy runs.** `heavy.yml` needed eleven runs to go green, in the private development repository
  (those runs are not public); earlier runs are
  `35065497824` to
  `35088397743`.
  Two green runs were not counted, because P10 had passed without really working: once at
  `words=4` (`35080734265`),
  once skipped before Whisper ran (`35088397743`).
  - **P10 on macOS**, three separate causes, each measured before it was fixed:
    - The speech engine writes AIFF-C whatever the file extension, so the output is now converted to WAV.
    - It writes asynchronously, so generation now waits until the audio is long enough and stable.
    - Whisper produced repeated-token noise on Apple's MPS backend, so it is pinned to CPU (`test_load_asr_model_pins_cpu_and_float32`).
  - **P12 on macOS:** the relevance gate now computes cosine similarity from the stored vectors itself. The diagnostic in run `35076304269` then showed that the store's scores and configuration were already correct. The failure comes from the score geometry on that platform, recorded as the limitation in A1.
- **2026-09-16, pre-publication review.** A security and privacy review of the tree, the full
  history and the pull-request descriptions added A8. The fixes that fit in the tree were
  made in the same change: interactive runs no longer overwrite committed outputs, Gradio
  analytics are off, CORS was removed from P09, the actions are pinned to commit SHAs, CI
  audits dependencies, and the test container runs unprivileged.
- **2026-09-18, new repository.** This repository was built from a rewritten history
  (A8), and the evidence above was re-run here: `ci.yml` and `heavy.yml` gave the same
  results as in the private repository, apart from one timing skip of P10 on macOS (A1).
- **Current state:** every item is ticked, and CI enforces A3 and the offline checks on
  every pull request. The repository is public since 2026-09-18.

## Publication record

Every step that changes what is public, or how the repository is protected, is recorded here
with its date and evidence, so the publication history stays visible in the repository
itself.

| Date | Step | Evidence |
|---|---|---|
| 2026-09-16 | Release 1.0.0 merged: twelve projects, one squash-merged pull request each. | `adc6e1f`, `bc73697` |
| 2026-09-16 | Real-LLM tests, LangChain 1.x, licence notices and security policy merged. | `6987934` |
| 2026-09-16 | Branch protection on `main` extended to administrators, with review conversations required to be resolved; Dependabot alerts and security updates enabled; wiki and projects disabled. | Repository settings, verified through the GitHub API |
| 2026-09-16 | Pre-publication security and privacy review; item A8 added. | This document, A8; `39a95b1` |
| 2026-09-16 | Hugging Face model revisions and container images pinned; trusted hosts for the local Flask servers. | `8044c99` |
| 2026-09-16 | Publication runbook, history-rewrite tooling and branch protection as code; dry run verified. | [publication.md](publication.md), A8; `e27fa56` |
| 2026-09-16 | Ollama 0.34.1 in compose and CI, real-LLM tests green on it. | `d9c3b33` |
| 2026-09-16 | Version 1.1.0 prepared: `pyproject.toml`, `CITATION.cff`, `CHANGELOG.md`. The tag and GitHub release follow publication. | [CHANGELOG.md](../CHANGELOG.md), `0267ba7` |
| 2026-09-18 | Private development repository renamed `genai-practice-projects-private`; it stays private and is archived after the release. | GitHub |
| 2026-09-18 | First build of this repository discarded before publication: a squash merge made on GitHub was signed with the author's institutional address, because the account's e-mail privacy setting was not yet on. That build was renamed away and kept private; the setting is now on, and this repository was rebuilt from the same history, with identical commit ids. | A8 |
| 2026-09-18 | This repository created, private, from the rewritten history: 26 commits, `verified: publishable`, gitleaks clean. | A8; `0267ba7` |
| 2026-09-18 | Settings applied: squash merge only, branch protection from `.github/branch-protection.json`, read-only workflow token, Dependabot alerts and security updates, description and topics. | Repository settings, verified through the GitHub API |
| 2026-09-18 | Ollama 0.34.2 in compose and CI; real-LLM tests green on it. | `626b3bb` (#2) |
| 2026-09-18 | Evidence moved to this repository; A8, C8 and C9 ticked. A fresh clone on Windows passed the core tests and ran 12/12 demos with a clean working tree. | Pull request #3; runs [35311363210](https://github.com/tudorandrian/genai-practice-projects/actions/runs/35311363210) and [35311307417](https://github.com/tudorandrian/genai-practice-projects/actions/runs/35311307417) |
| 2026-09-18 | Visibility changed to public; private vulnerability reporting, secret scanning with push protection, and approval for outside contributors' workflow runs enabled. C10 ticked. | Repository settings, verified through the GitHub API |
| 2026-09-18 | Full `release_check` with network access: `PASS links-resolve`. A7 ticked. | A7 |
| 2026-09-18 | Tag `v1.1.0` on the release commit `0267ba7` and GitHub release 1.1.0, with the 1.1.0 section of the changelog as notes. | [Release 1.1.0](https://github.com/tudorandrian/genai-practice-projects/releases/tag/v1.1.0) |
| 2026-09-18 | The four `chromadb` Dependabot alerts dismissed as not used: they affect only Chroma's HTTP server, which no project runs. | [SECURITY.md](../SECURITY.md) |
| 2026-09-18 | Private development repository and the discarded first build archived; both stay private. | GitHub |
