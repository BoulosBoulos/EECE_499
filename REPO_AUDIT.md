# Repository audit, 18 September 2026

Audited at commit `68aaf0627329e186e7300d17b20406d64749acc5` (tip of `main`),
18,657 tracked files. Everything below was checked against the repository, not
assumed.

Sections 1 to 4 and 6 were acted on in the `repo-cleanup` branch that carries
this file. Sections 5, 7 and 8 record findings that need no change or that
remain open; section 9 lists what is left.

---

## 1. Do not rewrite history

The paper cites commit `68aaf062` as the source of every number. Purging the
502 checkpoint files with `git filter-repo` or BFG would rewrite history and
**change that hash**, breaking the citation in a submitted manuscript. The
502 `.pt` files are what make the repository 4.3 GB, so the temptation is real.

**Done:** that commit is now tagged `paper-tiv-v1` and published as a release,
so the cited state is pinned independently of where `main` goes next. Any size
surgery should wait until after acceptance, with a Zenodo archive of the
original taken first.

---

## 2. The README contradicts the paper and the code

The existing README is 1,260 lines and genuinely good on structure. But five of
its technical claims are wrong, and one of them undercuts the paper's central
finding. All five were verified against the source.

| # | README says | The code says | Severity |
|---|---|---|---|
| 1 | §3: "no gradient from the PDE residual ever reaches the actor `π_θ`" | False for three of five methods. `soft_hjb_aux_agent.py:209`, `fusion_aux_agent.py:390` and `eikonal_aux_agent.py:562` each add a KL term directly to the actor loss. The README contradicts itself in §8.4, which describes that term. | **Critical.** The paper's headline finding is that this actor coupling causes the degradation. A reviewer who reads the repo first will conclude the paper's mechanism cannot exist. |
| 2 | §8.4: `L_actor_kl = λ·KL( π_θ ‖ π_soft )` | Forward KL, the other direction: `kl = (pi_s * (pi_s.log() - pi_t.log())).sum()` in `soft_hjb_aux_agent.py:208`. The paper says forward KL. | High. The direction is the mechanism. |
| 3 | §8.5: `c(xi) = max_a [‖F_a(xi) − xi‖ / dt]` | `residuals.py:_compute_v_eff_and_c_sq` computes `v_eff = [max_a v'_a · sigmoid((TTC'_a − TTC_thr)/0.5)] · clip(alpha_cz, 0.1, 1)`, clamped at `v_min`, with `c = 1/v_eff`. The TTC safety weight and the visibility factor are missing from the README. | High. Those two terms *are* the adaptation the paper claims as a contribution. |
| 4 | §8.7: Fusion's optimality critic uses the **Hard**-HJB residual | `fusion_aux_agent.py` docstring line 10 and its imports: **Soft**-HJB. | High. It changes which method Fusion is a fusion of. |
| 5 | §8.6: the CBF barrier is "the smooth minimum of three sub-barriers `h_stop`, `h_friction`, `h_ttc`" | Those identifiers appear nowhere in the codebase. `residuals.py:226` defines `h(xi) = U(xi) + cbf_safe_offset`. | Medium. Documents a design that was replaced. |

Minor: the README title says "Unsignalized Intersections" where the paper and
Prof. Daher's style rule use "unstructured"; §8.8's rule-based pseudocode has
an `elif` and an `else` that both return GO.

**Done:** the README in this branch is a corrected and extended rewrite. The
five claims above are fixed and the sections they sat in were rebuilt from the
source rather than patched.

---

## 3. A private file is published

`PromptE.txt` sat at the repository root, publicly readable, and held personal
and machine-specific material unrelated to the research: local filesystem
paths, hardware details, and correspondence about a hosting account. None of it
belongs in a repository named in the footnote of a journal submission.

**Done:** deleted in this branch, together with the editor backup file
`verification/CODE_REVIEW.md.save`. Deletion clears the tip only. Both files
remain retrievable from history, which is the second argument for a
post-acceptance history rewrite.

---

## 4. `.gitignore` says the opposite of what is tracked

`.gitignore` ignores `results/` and `*.pt`. The repository tracks **18,126
files under `results/`** and **502 `.pt` checkpoints**, all of them in
`results/ablation`. They were force-added or predate the ignore rules.

This is not harmful in itself, but it means the ignore file is now
misinformation: anyone who adds a new result will find it silently untracked
while thousands of older ones are tracked.

**Done:** the ignore rules are unchanged, because narrowing them risks pulling
gigabytes of new checkpoints into a commit by accident. Instead `.gitignore`
now carries a note stating that the tracked results are deliberate, why, and
how to force-add a new one.

The 502 checkpoints are the 4.3 GB. See section 1 before touching them.

---

## 5. Missing files a public research repository needs

| File | Status | Why it matters |
|---|---|---|
| `LICENSE` | added in this branch | Without one, the code is all-rights-reserved by default and nobody may legally use, fork or reproduce it. Prof. Daher's remark 4 asks for the repository to be shared with reviewers; sharing it unlicensed undercuts the point. MIT was chosen as the conventional default. **Confirm AUB's IP position before merging, and change it if the answer differs.** |
| `CITATION.cff` | added in this branch | GitHub renders a "Cite this repository" button from it. Its `version` and `commit` fields point at `paper-tiv-v1`. Its `license` field must track whatever `LICENSE` ends up saying. |
| `CONTRIBUTING.md` | missing | Optional for a thesis repository. |
| Pinned dependencies | **still unpinned** | `requirements.txt` uses `>=` throughout, so a fresh install today resolves different versions than the runs used. A `pip freeze` from the machine that produced the results, committed as `requirements-lock.txt`, would close the last reproducibility gap. This has to come from that machine, so it could not be done here. |

`requirements.txt` also lists `streamlit`, `plotly` and `optuna`, none of which
appear in the analysis pipeline. Worth checking whether they are still needed.

---

## 6. Cruft and duplication

| Item | Detail | Action |
|---|---|---|
| `verification/CODE_REVIEW.md.save` | editor backup file | **deleted in this branch** |
| Duplicate LaTeX tables | 16 filenames exist in **both** `results/tables/` and `results/analysis/tables/` (`main_results.tex`, `effect_sizes.tex`, `per_scenario_*.tex`, …) | keep `results/tables/`, which is what the paper cites; note the other as generated output |
| 155 `.log` files in `results/ablation` | training logs | keep if you want the audit trail, otherwise gzip |
| `.cursor/` and `.cursorignore` | IDE configuration | harmless, but `.cursor/commands/run-train.md` may contain local paths, worth a look |
| `tests/` contains only `__init__.py` | no tests, while `verification/` holds 323 files of actual checks including `test_residuals_math.py` and `test_smooth_clamp.py` | either move those into `tests/` or delete the empty package so the layout stops implying a test suite that is not there |
| `docs/` has 25 files with overlapping scope | `HYPERPARAMETERS.md` and `ABLATION_HYPERPARAMETERS.md`; `PDE_METHODS.md` and `PDE_METHODS_AND_ENV_UPDATES.md`; `SCENARIO.md` and `SCENARIO_SUMO.md`; `STATE.md` and `STATE_SCHEMA.md` | consolidate or add a `docs/INDEX.md` saying which is current |
| Stale branches | `results/cluster-2026-05-15`, `verification/2026-04-29` | delete if merged, or document why they exist |

---

## 7. Stale editorial marker inside the results, already handled

`results/analysis/PAPER_REPORT.md` contains pre-audit values that were later
falsified (a 19 per cent CBF divergence figure, a Fusion gap of 0.18 to 71.9).
A `TODO[DIAGNOSTICS PROVENANCE]` marker in the 23-page draft warned about
exactly this and said not to submit while it was unresolved.

**No action needed.** Commit `68aaf062` is itself the commit titled *"Mark
PAPER_REPORT.md superseded (pre-audit, contradicts current results)"*, and the
file already opens with a blockquoted `SUPERSEDED, DO NOT CITE` banner that
names both defect classes (episode-level pooling, and the two contradicted
claims) and maps each superseded section to its authoritative replacement under
`results/tables/`. That banner is more thorough than anything worth adding.

The post-audit values are correct in the data. I recomputed them and they match:
CBF residual grows in 77 of 224 runs (rate 0.3438, median late/early 0.5574);
the Fusion distillation gap grows in 225 of 240 runs from a median of 6.428 to
34.276. Those are the numbers now in the paper.

---

## 8. What is in good shape

Worth saying, because most of this repository is solid.

- **Every analysis table the paper needs is present**, all 15, with sane row
  counts.
- **Tier 1 is complete**: 12 cells × 6 methods, 59 of 60 augmented method-cells
  at ten seeds and one at nine, baseline at ten everywhere. That is the 719 runs.
- **The config lock works.** `config_lock.json` holds a SHA-256 of
  `config_frozen_v1.yaml` and `check_config_lock()` enforces it.
- **The code matches the paper** on every point I checked: `XI_DIM = 79`, the
  block layout of $\xi$, the nominal accelerations, the smooth clamp, the
  trapezoidal integration, $\lambda_{\mathrm{distill}} = 0.25$, the coupling
  channels per method, and the building margins.
- **The verification directory is substantial**: 323 files of preflight checks,
  determinism tests, audits and reconciliation notes.

---

## 9. Status

Done, in the tag and in this branch:

1. `68aaf062` tagged `paper-tiv-v1` and published as a release.
2. `PromptE.txt` and `verification/CODE_REVIEW.md.save` deleted.
3. The corrected `README.md` committed.
4. `LICENSE` and `CITATION.cff` added.
5. `.gitignore` annotated to explain the tracked results.

Open, and none of it blocks submission:

6. Commit `requirements-lock.txt` from the machine that produced the runs.
7. Consolidate `docs/`, resolve `tests/`, tidy the two stale branches.
8. Confirm AUB's IP position and adjust `LICENSE` and `CITATION.cff` if needed.
9. Leave the 4.3 GB alone until the paper is accepted.
