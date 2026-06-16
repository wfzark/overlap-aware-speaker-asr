export const meta = {
  name: 'entropy-audit-understand',
  description: 'Deep parallel investigation of the overlap-aware-speaker-asr workspace to ground an agentic-research-entropy audit',
  phases: [{ title: 'Investigate', detail: '6 parallel readers over harness, substance, ceremony, git, cleanup, feasibility' }],
}

const REPO = '/Users/a86198/Desktop/overlap-aware-speaker-asr'

const SCHEMA = {
  type: 'object',
  properties: {
    area: { type: 'string' },
    key_facts: { type: 'array', items: { type: 'string' }, description: 'Concrete facts with numbers and file paths' },
    constraints_or_risks: { type: 'array', items: { type: 'string' } },
    runnable: { type: 'array', items: { type: 'string' }, description: 'Exact commands or python invocations confirmed to work' },
    examples: { type: 'array', items: { type: 'string' }, description: 'Representative file refs each with a one-line description of what it actually does' },
  },
  required: ['area', 'key_facts'],
}

phase('Investigate')

const tasks = [
  {
    label: 'harness-mechanics',
    prompt: `Repo: ${REPO}. You are investigating the development HARNESS so a new advisory check can be added SAFELY without breaking gates.
Read: scripts/harness/quality.py, scripts/harness/contract_check.py, scripts/harness/install_hooks.py, everything under .githooks/, .github/workflows/contract-guard.yml, .github/workflows/test.yml, .github/PULL_REQUEST_TEMPLATE.md, docs/harness/README.md, docs/harness/workflow_spec.md (if present), src/project_harness.py, and the Makefile.
Answer precisely:
1. How does the GitNexus contract decide pass vs fail? What exactly is "critical code" (router-core, evaluation-core, harness) and what is the precise paired-test rule (which test path must accompany which src path)?
2. What runs in pre-commit vs pre-push vs CI? Which gates are BLOCKING vs ADVISORY? How is "advisory" implemented (exit code 0 + warn)?
3. Exactly how could I add a NEW advisory signal (a function/CLI that prints a warning but never fails the gate) — what file would it live in and how is it wired? Show the smallest safe integration point.
4. How to run the test suite, and how to run only ONE test file fast. What does SKIP_QUALITY_HOOKS do?
5. Anything that would make my change accidentally fail the contract (e.g., adding src/ file without a paired test).
Return concrete file paths + line references. Do NOT modify anything.`,
  },
  {
    label: 'substance-core-and-data',
    prompt: `Repo: ${REPO}. Inventory the GENUINE research substance (the opposite of ceremony) so I can ground a substance-vs-ceremony classifier in real examples.
Read at a high level these substantive modules (skim each, ~what it computes): src/adaptive_router.py, src/adaptive_router_v2.py, src/evaluate_cer.py, src/evaluate_speaker_cer.py, src/evaluate_cpcer_lite.py, src/evaluate_error_types.py, src/risk_aware_selector.py, src/compute_aware_cascade.py, src/cascade_tiers.py, src/separation_phase_diagram.py (CONFIRM it exists; if so what does it output and does it show a non-monotonic separation gain?), src/generate_synthetic_overlap.py, src/run_experiment.py, src/config.py, src/io_helpers.py.
Read the REAL (non-ceremony) result data: results/tables/adaptive_routing_results.csv (full), results/tables/audio_manifest.csv (head), any synthetic routing CSV, and results/figures/curated/current_results_summary.md, results/figures/curated/best_method_by_case.md, results/figures/curated/synthetic_routing_summary.md.
Answer:
1. Confirm the per-case CER numbers (mixed/separated/separated_cleaned) for all 5 gold cases and the overlap_level ordering. Is separation gain non-monotonic in overlap level? Quote the exact numbers.
2. Does src/separation_phase_diagram.py exist and produce results/tables/separation_phase_diagram.csv + a figure? If yes, summarize what it shows. If not, say so.
3. Give 8-10 files that are unambiguously SUBSTANCE and for EACH a one-line reason (computes a metric / transforms audio / fits a model / produces a falsifiable number) vs files that merely emit a status string.
4. How many distinct REAL result CSVs exist in results/tables vs how many are "_coordination_/_writeback_/_checklist_/_receipt_" ceremony CSVs (give counts via the file names you see).
Do NOT modify anything.`,
  },
  {
    label: 'ceremony-anatomy',
    prompt: `Repo: ${REPO}. Characterize the CEREMONY corpus rigorously. The repo has ~800 src files whose names contain writeback/wave/handoff/receipt/bridge_checklist/coordination/operator_brief/runbook_card/milestone_card/completion_summary/presentation/storyboard/walkthrough.
Fully READ 7-9 representative ceremony files spanning the types, e.g.: src/demo_storyboard_review_pass_advance_bridge_checklist.py, src/frontier_operator_next_action_status_handoff_completion_summary.py, src/meeteval_cpwer_execution_status_batch_handoff_completion_summary_handoff.py, src/cascade_benchmark_evidence_receipt_coordination_writeback.py, src/speaker_profile_embedding_trial_execution_scaffold_completion_summary.py, src/demo_wave100_presentation_writeback.py, plus any one *runbook_card* or *operator_brief* you find.
For EACH file determine: does it (a) compute/transform/evaluate any real data or model, or (b) merely assemble hardcoded/templated strings and write a markdown/CSV "status/handoff/receipt" file that mostly references OTHER ceremony files? Note line counts and whether the "output" is just another status doc.
Answer:
1. A precise taxonomy of ceremony types and what each does (with the representative file + line refs).
2. What fraction of the ones you read are pure-string-emission with no data computation? Quote evidence (e.g., the function bodies just build a dict of strings and write .md).
3. The self-reference pattern: do ceremony files/outputs primarily cite other ceremony files? Give 3 concrete citation chains you observed (A references B references C).
4. The single most damning concrete example (file + what it does) that proves these are non-substantive.
Do NOT modify anything.`,
  },
  {
    label: 'git-dynamics',
    prompt: `Repo: ${REPO}. Produce a precise quantitative TIMELINE of substance vs ceremony accumulation from git history. Use git commands (read-only).
Define CEREMONY filename regex = writeback|wave|handoff|receipt|bridge_checklist|coordination|operator_brief|runbook|milestone|completion_summary|presentation|storyboard|walkthrough|go_no_go|queue_status|phase_checkpoint|next_action.
1. For src/*.py ADD events (git log --all --diff-filter=A --name-only --format with %ad --date=short), produce a per-day table: date, ceremony_adds, substance_adds (substance = .py adds NOT matching the regex). Cover the whole history.
2. Do the same aggregated for tests/ and for results/tables/.
3. Count total commits; count commits whose message contains wave/writeback/handoff/receipt/presentation/coordination vs total. List 10 representative commit subjects from the Jun 12-13 explosion window.
4. Enumerate the frontier/wave* branches (git branch -a | grep wave) and count them; note the max wave number you can find anywhere (branches, filenames, commit msgs).
5. Authorship: src/*.py adds per author.
Return clean tables (as text) and the key inflection date. Do NOT modify anything.`,
  },
  {
    label: 'existing-cleanup',
    prompt: `Repo: ${REPO}. Understand the EXISTING cleanup/archival effort so my new audit ALIGNS with it rather than conflicts.
Read: docs/archive-plan.md, docs/branch-audit.md, docs/readability-cleanup-baseline.md, docs/readability-cleanup-manifest.md, the HEAD (first ~40 lines) of docs/readability-cleanup-moved-files.tsv, docs/README.md, docs/maintenance_harness.md, docs/repo_evolver.md, CONTRIBUTIONS.md, AGENTS.md. Also list results/figures/archive/ structure (find -maxdepth 2) and count how many files have already been archived vs how many ceremony files still live in src/ and tests/.
Answer:
1. What taxonomy/labels do the maintainers already use for ceremony (wave records, receipts, writebacks, bridge checklists, etc.)? I want to reuse their vocabulary.
2. What has ALREADY been moved/archived (counts + locations) vs what remains live in src/ and tests/ and results/tables/?
3. Is there an existing policy against deleting (archive-don't-delete)? Quote it.
4. Does any existing doc already quantify the ceremony problem with numbers? If so, what numbers (so I don't duplicate, I extend)?
5. Any naming or output-path conventions I must follow for a NEW experimental analysis (the charter wants new experimental result dirs, clear labels).
Do NOT modify anything.`,
  },
  {
    label: 'run-feasibility',
    prompt: `Repo: ${REPO}. Determine EXACTLY how to run python code and tests here, without GPU.
Investigate both virtualenvs: ${REPO}/.venv and ${REPO}/.venv-external. For EACH, run its python and check which packages import: numpy, scipy, soundfile, yaml(PyYAML), matplotlib, whisper. Report the exact interpreter path that has numpy+matplotlib+PyYAML (needed for analysis/plotting) — I need a deterministic, GPU-free, network-free way to run analysis code and tests.
Then:
1. Run ONE substantive, fast test to confirm green, e.g.: <interp> -m unittest tests.test_evaluate_cer -v  (or test_adaptive_router*, test_cascade_tiers — pick one that does NOT need whisper/audio). Report pass/fail and runtime.
2. Confirm whether 'git log' and plain file IO work (they will) — these are all my tool needs.
3. Identify any test or module that requires whisper/torch/real audio (so I AVOID depending on them).
4. Report the exact command to run a single new test file under tests/ with unittest, and confirm matplotlib can render to a PNG headless (Agg backend) by actually doing a tiny savefig test with the chosen interpreter and reporting success.
Return the precise interpreter path + confirmed commands. You MAY run read-only/innocuous commands and create a throwaway file in /tmp only. Do NOT modify the repo.`,
  },
]

const findings = await parallel(tasks.map(t => () =>
  agent(t.prompt, { label: t.label, phase: 'Investigate', schema: SCHEMA })
    .then(r => ({ label: t.label, ...r }))
))

return findings.filter(Boolean)
