# Hugging Face real-photo pairing pipeline

## Scope

`specs/v2.json` defines the default paired-data workflow:

```text
pinned Hugging Face real photograph
  → deterministic Moondream caption
  → content-only normalized caption
  → one frozen generation prompt
  → two candidates for every synthetic slot
  → deterministic first passing candidate
  → auditable Parquet package
```

The research claim is **real_photo versus synthetic**. A Hugging Face row is
not described as an unedited camera original unless independent capture
evidence exists. EXIF is not required and its absence is not interpreted as
evidence either way.

`specs/v1.json` remains available for reproducing the earlier
Wikimedia/camera-evidence workflow. V2 does not change the meaning of v1.

## Checked-in pilot decisions

The three-group v2 pilot pins:

| Component | Repository | Immutable revision |
|---|---|---|
| Real source | `Spawning/PD12M` | `867988b01138799b89d3ffdd5b4f7e1455951f32` |
| Captioner | `vikhyatk/moondream2` | `9a7d4024050840e001defacec2b00727e89149e6` |
| Alignment QA | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |

PD12M is used because each selected row exposes a stable ID, source URL,
dimensions, MIME type, and per-image public-domain/CC0 declaration. The
dataset-level metadata license and the per-image artifact license are distinct:
the builder accepts only rows whose `license` value normalizes to `CC0-1.0`.
The byte source is the pinned row's `url`; it is preserved unchanged.

PD12M contains artwork, scans, specimens, and other non-photographic material
in addition to photographs. V2 applies three screens before assigning accepted
source positions:

1. metadata format, dimensions, license, and obvious non-photo term filters;
2. corrupt/blank image checks;
3. pinned CLIP photo-versus-non-photo and safety gates.

The source dataset's existing caption is used only by the coarse metadata
rejection screen. It is never used as the frozen caption or generation prompt.

Moondream runs locally with `length="short"` and `temperature=0`. The exact raw
caption and the normalized caption are both retained. `content-only-v1`
removes presentation scaffolding, aesthetic judgments, hedging, and unsupported
camera/style language while retaining visible subjects, actions, relationships,
settings, and attributes.

The CLIP thresholds in `specs/v2.json` are fixed pilot gates:

- image–caption cosine similarity: at least `0.20`;
- photographic-class probability: at least `0.55`;
- unsafe-class probability: at most `0.35`.

They make the pilot deterministic; they are not universal quality guarantees.
Before a larger release, calibrate them on a labeled sample from every real
source and generator family, record the calibration set and operating point,
and version `threshold_policy`.

## Install

ImageMagick 7 must provide `magick`. Install all build dependencies with:

```bash
./install.sh
```

The source, captioning, alignment, and generation extras can also be installed
separately:

```bash
python3 -m pip install -e '.[source,captioning,alignment,generation,parquet]'
```

The first non-dry build downloads the pinned Moondream, CLIP, FLUX, SDXL, and
SD 1.5 weights through Hugging Face. Device choice is CUDA, then MPS, then CPU.
CPU is supported but a complete build is slow.

## Dry run

Dry run validates the complete contract and reports work without loading the HF
dataset, downloading bytes or weights, running inference, or writing output:

```bash
python3 -m mai.dataset_cli build \
  --spec specs/v2.json \
  --output .mai-data/v2-package \
  --cache .mai-data/v2-cache \
  --dry-run
```

The checked-in pilot reports:

- 3 source-photo groups;
- 21 accepted-package slots if no group is quarantined;
- 3 captions;
- 36 generation candidates;
- 42 QA calls, including source screens and caption alignment;
- no manual review decisions.

The cache path is read only to report reusable work during dry run. A missing
cache is not created.

## Build

Run the pilot with:

```bash
python3 -m mai.dataset_cli build \
  --spec specs/v2.json \
  --output .mai-data/v2-package \
  --cache .mai-data/v2-cache
```

The builder processes source rows in the deterministic order produced by the
pinned dataset, `sample_seed`, and streaming shuffle buffer. A group
`source_index` means the Nth source row that passes license, metadata, image,
photo/safety, exact-duplicate, and perceptual-near-duplicate gates. Processing
all earlier positions makes a partial build select the same row as a full build.

The split is stored on the planned group before captioning or generation.
Exact SHA-256 and 64-bit difference hashes are checked before positions are
assigned, so accepted duplicates cannot cross train, validation, and test.
All descendants of one real photo inherit that group and split.

For each accepted real photo:

1. Preserve the downloaded bytes and record SHA-256 and byte count.
2. Cache the deterministic raw Moondream caption by source checksum and pinned
   caption configuration.
3. normalize the caption under `content-only-v1`;
4. render and freeze the single prompt under `frozen-caption-v1`;
5. score the real image against that normalized caption;
6. generate two independently seeded candidates for every configured slot;
7. apply the same image–caption score and the same photo/safety gates to every
   candidate;
8. select the first candidate by index that passes all gates.

Candidate scores are never used to rank passing candidates. A later,
higher-scoring candidate cannot replace an earlier passing candidate. This
avoids aesthetic best-of-N selection.

## Quarantine and human audit

Caption exceptions, real-image QA failures, generator exceptions, corrupt
outputs, and all-candidate failures quarantine the complete source-photo group.
No partial sibling group enters the package.

Quarantine receipts are written under:

```text
.mai-data/v2-cache/quarantine/<semantic-group-id>.json
```

They include the failed stage, reason, source lineage where available, candidate
indices, seeds, scores, and exceptions. Receipts for the current build are
copied to `package/quarantine/` and checksum-indexed in `dataset.json`.

Human review is not required for every generated candidate. A deterministic 5%
group sample is marked `human_audit_required`; its status starts as `pending`.
All quarantined groups and any future manual override require review outside the
automatic acceptance path. Manual audit metadata may be added without changing
the frozen caption, source lineage, or automatic first-passing result.

## Required lineage

The real-photo receipt retains:

- HF dataset ID, exact revision, source split, row ID, and source URL;
- declared and normalized artifact license;
- byte-identical original SHA-256 and byte count;
- dimensions, format, color channels, blank-image statistic, and perceptual
  hash;
- raw and normalized Moondream captions;
- caption policy, model ID, exact revision, settings, runtime, and device;
- QA model ID, exact revision, thresholds, runtime, and real-image scores;
- source-photo group, split, and audit-sample status.

Every synthetic receipt additionally retains:

- the same source-photo and caption lineage;
- generator family, model ID, exact revision, settings, runtime, scheduler, and
  seed;
- every candidate's index, checksum, byte count, health result, scores, and
  failure reasons;
- selected candidate index and `automatic-first-passing-v1`;
- quarantine and manual-override status.

The package validator fails missing or mismatched revisions, captions, source
rows, source checksums, lineage, QA models, candidate arrays, or first-passing
selection.

## Package layout

Accepted groups use the existing Parquet-native package contract:

```text
package/
├── README.md
├── dataset.json
├── groups.json
├── validation_report.json
├── data/<split>-<shard>.parquet
├── originals/
│   ├── real_photo/real_photo/*
│   └── synthetic/<generator-family>/*
├── receipts/<sample-id>.json
└── quarantine/<semantic-group-id>.json
```

Originals are copied byte-for-byte. All real and synthetic images then undergo
the same deterministic 512×512 RGB normalization and metadata stripping before
their normalized bytes are embedded in Parquet. A group is never split across
Parquet shards.

Validate independently with:

```bash
python3 -m mai.dataset_cli validate \
  --package .mai-data/v2-package
```

Publish only after validation:

```bash
hf auth login
python3 -m mai.dataset_cli publish \
  --package .mai-data/v2-package \
  --repo-id OWNER/mai-v2-pilot \
  --tag v2-pilot
```

Pin the returned Hub commit in every analysis.

## Offline tests

The tests replace HF streaming, Moondream, CLIP, and all generators with local
deterministic adapters. They perform no network access and cover:

- immutable revision validation;
- deterministic source selection and cache keys;
- caption normalization;
- exact and perceptual duplicate rejection;
- identical real/generated semantic thresholds;
- automatic first-passing selection;
- exception and all-candidate quarantine;
- source/caption lineage equality;
- a complete three-group, 21-sample package build.

Run:

```bash
python3 -m unittest discover -s tests
```

## Research safeguards and limitations

- Treat the source-photo group—not generated siblings—as the statistical unit.
- Keep every sibling and transformed derivative in its source group's split.
- Weight source groups and generator families equally in analysis.
- Do not treat repeated candidates or sibling slots as independent samples.
- Preserve byte-identical originals and apply one normalization implementation
  to both origin classes.
- Expand beyond PD12M before the full release; use several independent,
  license-auditable real-photo sources.
- Maintain a holdout with a different real source, a second caption policy or
  human captions, and unseen generator families.
- Run a small Moondream-versus-alternative-captioner ablation. Captioner errors
  can create or suppress the apparent real/synthetic geometry.
- Audit CLIP thresholds for demographic, cultural, geographic, and content
  biases. CLIP scores are screening signals, not factual or safety proofs.
- Do not revive the strict `camera` claim without defensible capture
  provenance.
