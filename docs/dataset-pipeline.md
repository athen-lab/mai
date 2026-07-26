# Hugging Face dataset pipeline

## What Build does

`specs/v1.json` is the checked-in experiment contract. It contains all 200
semantic groups, frozen prompts, group-locked splits, the seven-slot sample
matrix, camera acquisition policy, and generator configurations.

For the selected groups, Build runs this sequence:

1. Validate the complete design before downloading inputs or model weights.
2. Search Wikimedia Commons for semantically matched JPEGs.
3. Require camera make, camera model, capture time, a compatible license, and
   no detected editor name in EXIF `Software`.
4. Download each original photograph and record its source page, license,
   search query, Commons identifier, EXIF, dimensions, and checksum.
5. Reject duplicate camera records or bytes before local generation starts.
6. Run the same frozen prompt through every configured local model slot.
7. Cache the acquired and generated inputs with checksummed receipts.
8. Retain byte-identical originals and create deterministic 512×512 RGB PNG
   analysis versions with ImageMagick.
9. Validate the complete group matrix and write a local Hugging Face package.

No separate cache-population step is required. The cache is an internal,
reusable stage of Build. A prompt, slot, seed, or model-configuration change
produces a different cache key; a matching entry is reused only when its
SHA-256 checksum still matches its receipt. A cached-only rebuild neither
loads a model nor runs inference.

Build does **not** stream records directly into Hugging Face. It finishes and
validates `.mai-data/package` locally first. Publish is a separate operation so
an incomplete run cannot become a release.

## Install and authenticate

ImageMagick 7 must provide the `magick` command. Install the Python generation
and Hub clients with:

```bash
./install.sh
```

The v1 generation matrix is entirely local: FLUX.1-schnell, Stable Diffusion
XL 1.0, and Stable Diffusion 1.5 run through Diffusers. No image-generation API
or paid provider is used. On first use, Hugging Face downloads model weights
into its normal local cache. `hf auth login` may be needed to accept a model
license, and the same login is used when publishing the finished dataset.

Device selection is automatic: CUDA first, then Apple MPS, then CPU. CUDA uses
model CPU offload by default to reduce VRAM pressure. Only one model family is
kept loaded at a time. CPU generation is supported but can be very slow,
especially for FLUX.

## Smoke test and real run

Start with a network- and generation-free dry run:

```bash
python3 -m mai.dataset_cli build \
  --spec specs/v1.json \
  --output .mai-data/package \
  --cache .mai-data/cache \
  --group-id animals-001 \
  --group-id architecture-001 \
  --group-id food-005 \
  --dry-run
```

That selection covers three content categories and the complete planned axis:
3 camera photographs plus 18 synthetic outputs, or 21 samples total. Dry run
validates the selection and reports cache hits, camera downloads, and local
generation jobs without accessing the network or writing files.

Run the same selection for real:

```bash
python3 -m mai.dataset_cli build \
  --spec specs/v1.json \
  --output .mai-data/package \
  --cache .mai-data/cache \
  --group-id animals-001 \
  --group-id architecture-001 \
  --group-id food-005
```

The TUI exposes the same arguments through `python3 -m mai`; select `Build
dataset`, choose exact groups in the popup, and run. The running screen shows
each acquisition/generation cache hit or local job, then each normalization
step.

For the 200-group preliminary run, omit `--group-id`:

```bash
python3 -m mai.dataset_cli build \
  --spec specs/v1.json \
  --output .mai-data/package \
  --cache .mai-data/cache
```

The full v1 matrix is 1,400 images: one camera image plus two outputs from each
of three generator families for every group. An unfiltered build fails if the
spec contains fewer than `dataset.target_group_count` groups.

## Camera provenance and curation

The default Wikimedia adapter is an automated acquisition screen, not proof
that an image has never been edited. It rejects missing camera EXIF, missing
license data, non-JPEG files, images below 512 pixels on either side, and common
editor names such as Photoshop or Lightroom in EXIF `Software`. The original
file and all returned evidence remain available for audit.

Before treating a release as a strict camera-origin benchmark, visually review
each selected photograph, its Commons source page, and its receipt. To lock a
reviewed source, add a `camera_source` object to that group:

```json
{
  "camera_source": {
    "adapter": "direct-url",
    "url": "https://example.org/original.jpg",
    "sha256": "64-lowercase-hex-characters",
    "collection_id": "collection-name",
    "source_record_id": "stable-record-id",
    "landing_page_url": "https://example.org/record",
    "license": {
      "name": "CC BY 4.0",
      "url": "https://creativecommons.org/licenses/by/4.0/"
    },
    "capture": {
      "camera_make": "Canon",
      "camera_model": "Canon EOS 5D Mark IV",
      "captured_at": "2024-01-01T12:00:00Z"
    }
  }
}
```

The direct adapter requires a JPEG, verifies its declared SHA-256 before use,
and records the immutable URL and source metadata. For Wikimedia acquisition,
`camera_source: {"query": "custom search terms"}` can override the query without
changing adapters.

Hybrid, image-to-image, edited, rephotographed, or otherwise ambiguous samples
must not be marked `in_scope`. The initial validator accepts only
`{"in_scope": true, "ambiguity_flags": []}`.

## Build specification

The root object has:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Must be `2.0.0`. |
| `dataset` | object | Dataset-wide design and providers. |
| `groups` | array | Preferred explicit semantic-group catalog. |
| `samples` | array | Legacy manual flat samples; used instead of `groups`. |
| `samples_file` | string | Legacy manual JSONL path; used instead of `groups`. |

### Dataset fields

| Field | Type | Meaning |
|---|---|---|
| `dataset_id` | ID | Stable lowercase dataset identity. |
| `title` | string | Dataset-card title. |
| `description` | string | Scope summary. |
| `license` | string | Dataset policy; each sample also records its own license. |
| `target_group_count` | integer | Required count for an unfiltered build. |
| `seed_base` | integer | Base for deterministic per-group, per-slot seeds. |
| `expected_slots` | array | Complete sample matrix required in every group. |
| `camera_acquisition` | object | Default camera adapter and settings. |
| `generators` | object | Generator-family configurations keyed by family ID. |
| `normalization` | object | Optional normalization-profile overrides. |

There must be exactly one `camera` slot and at least two synthetic generator
families. Every synthetic slot has a unique `slot_id` and a
`generator_family`. Repetitions are explicit slots, not a runtime “seeded
groups” option.

### Generator fields

| Field | Type | Meaning |
|---|---|---|
| `family_id` | ID | Must equal the key in `dataset.generators`. |
| `adapter` | ID | `local-diffusers`. |
| `model_id` | string | Hugging Face model repository identifier. |
| `model_revision` | string | Optional requested revision; the resolved commit is recorded. |
| `device` | string | `auto`, `cpu`, `mps`, `cuda`, or an explicit CUDA device. |
| `cpu_offload` | boolean | Use Diffusers model CPU offload on CUDA; default `true`. |
| `settings` | object | Resolution, inference steps, guidance, and optional negative prompt. |
| `output_terms_url` | URL | Terms governing generated outputs. |

Every local slot receives a deterministic 31-bit seed derived from `seed_base`,
group ID, and slot ID. The receipt records that seed, the resolved model
repository commit, Diffusers pipeline class, execution device, and Torch dtype.

### Group fields

| Field | Type | Meaning |
|---|---|---|
| `semantic_group_id` | ID | Stable group identity. |
| `content_category` | string | Controlled content stratum. |
| `split` | ID | Group-locked train, validation, or test split. |
| `prompt` | object | `prompt_id`, exact `text`, and `frozen: true`. |
| `camera_source` | object | Optional query override or pinned direct source. |
| `samples` | array | Optional legacy manual records; absent means on demand. |

When `samples` is absent, Build creates every expected slot automatically.
When it is present and nonempty, the group is treated as a fully manual group
and must define its entire slot matrix.

### Prepared sample fields

The builder records these fields internally and in the package metadata:

| Field | Meaning |
|---|---|
| `sample_id` | Unique group-plus-slot identity. |
| `semantic_group_id` | Parent semantic group. |
| `slot_id` / `origin_class` | Experimental slot and camera/synthetic class. |
| `prompt` | Frozen prompt identifier and exact text. |
| `input_path` | Internal cached input used for the local build. |
| `source` | Collection, record ID, landing page, and license. |
| `capture` | Camera make/model/time and automated edit-screen result. |
| `generation` | Family, model, revision, provider, settings, and seed status. |
| `scope` | In-scope decision and ambiguity flags. |
| `provenance` | Acquisition or generation receipt. |
| `audit` | Selection method, review information, and cache status. |

IDs use lowercase letters, numbers, dots, underscores, and hyphens.

## Local package and Hub release

The ignored `.mai-data/` directory is the default local workspace:

```text
.mai-data/
├── cache/
│   ├── assets/                    acquired/generated inputs
│   └── receipts/                  checksummed cache receipts
└── package/
    ├── README.md                  Hugging Face dataset card
    ├── dataset.json               release/model contract and spec checksum
    ├── groups.json                compact group index
    ├── validation_report.json     integrity/design audit
    ├── data/<split>/
    │   ├── images/*.png           normalized analysis images
    │   └── metadata.jsonl         ImageFolder sample table
    ├── originals/<origin>/*       byte-identical originals
    └── receipts/*.json            per-sample provenance evidence
```

`metadata.jsonl` replaces the old `manifest.jsonl`, `acquisition.jsonl`, and
prompt table. A long metadata table is expected; Hugging Face versions it with
the dataset.

Validate and publish only after Build succeeds:

```bash
python3 -m mai.dataset_cli validate --package .mai-data/package
hf auth login
python3 -m mai.dataset_cli publish \
  --package .mai-data/package \
  --repo-id OWNER/mai-pilot \
  --tag v1
```

Record the returned commit SHA and pin it in experiments:

```python
from datasets import load_dataset

dataset = load_dataset("OWNER/mai-pilot", revision="COMMIT_SHA")
```

To retrieve exact groups later:

```bash
python3 -m mai.dataset_cli pull \
  --repo-id OWNER/mai-pilot \
  --revision COMMIT_SHA_OR_TAG \
  --output .mai-data/selected \
  --group-id animals-001
```

## Validation guarantees

The builder and validator enforce:

- a complete expected-slot matrix in every selected group;
- one frozen prompt, category, and split per group;
- camera evidence and text-to-image-only synthetic receipts;
- explicit in-scope status with no ambiguity flags;
- byte-identical originals with SHA-256 and byte counts;
- deterministic 512×512, 8-bit RGB PNG normalized images without metadata;
- unique original and normalized image content;
- a group index equal to the unified metadata.

These relationships support group-locked sampling, PCA/UMAP, distribution
distances, linear probes, k-NN, cross-generator transfer, transformations, and
stability analysis of a learned AI direction.

## Complete CLI reference

The TUI is the human entrypoint. `python3 -m mai.dataset_cli` is the automation
surface.

### `init`

| Parameter | Meaning |
|---|---|
| `--spec PATH` | New empty build-spec path. |
| `--force` | Replace an existing spec. |

The checked-in `specs/v1.json` is already suitable for smoke and real runs;
`init` is needed only for a genuinely different experiment design.

### `build`

| Parameter | Meaning |
|---|---|
| `--spec PATH` | Group catalog and model contract. |
| `--output DIR` | Local Hugging Face package destination. |
| `--cache DIR` | Reusable acquisition/generation cache; defaults to `cache` beside output. |
| `--group-id ID` | Exact group to build; repeat for more. Omit for all groups. |
| `--force` | Replace an existing recognized package, never the cache. |
| `--dry-run` | Validate and report counts without network access or writes. |

### `validate`

| Parameter | Meaning |
|---|---|
| `--package DIR` | Local package containing `dataset.json`. |
| `--report PATH` | Optional additional JSON report destination. |

### `publish`

| Parameter | Meaning |
|---|---|
| `--package DIR` | Validated local package. |
| `--repo-id OWNER/NAME` | Hugging Face dataset repository. |
| `--revision NAME` | Target branch; default `main`. |
| `--tag NAME` | Optional immutable release tag after upload. |
| `--private` | Create a private repository when new. |
| `--commit-message TEXT` | Custom Hub commit message. |

### `groups`

| Parameter | Meaning |
|---|---|
| `--repo-id OWNER/NAME` | Hugging Face dataset repository. |
| `--revision SHA_OR_TAG` | Pinned release revision. |
| `--json` | Print full group index instead of a table. |

### `pull`

| Parameter | Meaning |
|---|---|
| `--repo-id OWNER/NAME` | Hugging Face dataset repository. |
| `--revision SHA_OR_TAG` | Commit or release tag to retrieve. |
| `--output DIR` | Verified local subset destination. |
| `--group-id ID` | Exact group to retrieve; repeat for more. |
| `--force` | Replace an existing recognized destination. |
