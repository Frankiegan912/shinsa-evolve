# Public release checklist

## Approved release decisions

- [x] License code and the historical data bundle under Apache-2.0.
- [x] Use protocol v3 with Codex CLI 0.146.0, `gpt-5.6-sol`, low reasoning, and a prompt hash.
- [x] Publish historical and protocol-v3 data as separate checksummed Release assets, not in Git.
- [x] Publish protocol-v4 fixed-raw data as a third, separate checksummed Release asset.
- [x] Use the replicated LunarLander studies as primary evidence; label Breakout as a historical pilot.

## Automated gates

- [x] Current tree and Git history scanned for common secret formats and restricted keywords.
- [x] Public candidate contains no restricted-project association terms or personal absolute paths.
- [x] README labels the report as not peer reviewed and discloses LLM assistance.
- [x] Citation metadata, author email, license, and repository URL are present.
- [x] Historical headline statistics frozen and independently recomputable.
- [x] Historical evidence trees have deterministic SHA-256 digests.
- [x] New smoke runs use fresh directories and make no LLM calls.
- [x] New archives save audit seeds, per-episode returns, training seeds, and history.
- [x] Resume rejects changed configuration and missing RNG state.
- [x] Continuous and interrupted fake runs produce the same parent sequence.
- [x] LLM runs require an explicit model and record CLI/model identity.
- [x] Build and inspect the licensed, sanitized release bundle and checksum.
- [x] Clean-copy wheel build/import, offline smoke, and paper compile using locked build tools.
- [x] Rebuild every released table and figure from only the sanitized bundle plus public code.
- [x] Recompute the fixed-raw summary exactly from its sanitized six-run bundle.
- [x] Rebuild the final nine-page paper with Tectonic and visually inspect every rendered page.

## Excluded from the public bundle

- `runs/mac_batch.log` and other machine-local logs;
- smoke runs and temporary verification outputs;
- `.venv/`, caches, bytecode, egg-info, `.DS_Store`, and `.obsidian/`;
- ignored `implementation-notes.md`;
- all external restricted-source materials.

The historical builder copies only the six named historical runs, root seed recipes, and frozen
historical evidence report. The protocol-v3 builder copies the six replicated LunarLander runs,
including recipes, parameters, audit seeds/returns, RNG checkpoints, and the primary summary.
The protocol-v4 builder copies only the six successful fixed-raw runs and explicitly excludes the
failed first seed-101 attempt. All builders sanitize `seed_recipe` paths, exclude logs and caches,
rescan text outputs, emit per-file hashes in `BUNDLE_MANIFEST.json`, and produce deterministic
`.tar.gz` archives with sibling `.sha256` files.
