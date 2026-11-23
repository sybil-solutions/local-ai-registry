# Evidence reproduction

Extract the three tar.gz archives into this directory with a standard tar tool.
They preserve runs/, qualification/, execution/, and panel-review/ paths. Run data
contains results, enriched full-output ledgers, launch and resolved profiles;
telemetry/logs are separate. Duplicate samples.jsonl files are omitted.

Top-level core.py reads configs/base-launch.json instead of a temporary registry
checkout and omits unused exploratory launch/stop/CLI helpers; use runner.py or
qualification.py for execution. Original source snapshots remain archival evidence.
All literal private home directories become /home/USER, which must be
replaced for replaying saved launch JSON. Prefer the supplied runner, which derives
the Hugging Face host cache from Path.home(). Qualification CLI --daily-root and
--smoke-source can point at the included frozen daily-repo and smoke-original.py.
Frozen and execution sources are archival evidence and retain their original
registry dependency; use top-level portable sources for a new run.

package-provenance.json records original file SHA256 and published content SHA256
with every adaptation. Existing execution hashes in results and source manifests
remain original: this bundle does not claim adapted files are the executed bytes.
New runs naturally produce different source hashes. Gzip/tar metadata is deterministic.
SHA256SUMS.json hashes stored artifacts, including compressed files, excluding itself.

Measured copy/GEMM hardware microbenchmarks, conditional traffic theory, exploratory
failures, controlled A/B runs, and qualification are distinct evidence sources.
Do not describe theoretical traffic scenarios as an achieved percentage of a
practical ceiling. NVFP4 grouped-MoE prefill has no measured hardware ceiling here.
Final qualification and adoption claims require the separately reviewed results.
