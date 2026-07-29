# Multilayer Authenticity Identifier (MAI)

MAI studies the representation geometry of license-auditable real photographs
and fully generated photorealistic images. The immediate goal is an
embedding atlas across encoders, layers, generator families, and controlled
transformations—not another production detector.

This Git repository is the dataset **control plane**: it contains the workbench,
build/validation/publishing code, documentation, and tests. Images and release
metadata are the **data plane** and belong in a Hugging Face dataset repository.
The current paired-photo pilot is checked in at `specs/v2.json`;
`specs/v1.json` remains reproducible as the earlier camera-evidence protocol.
Local inputs and build directories live under ignored `.mai-data/`.

## Start here

ImageMagick 7 is required for deterministic normalization. Install the
generation, Parquet, and Hub clients, then start the workbench:

```bash
./install.sh
hf auth login
python3 -m mai
```

With Nix, enter the pinned development shell first. It creates and activates
an ignored `.venv` with Python 3.12:

```bash
nix develop "path:$PWD"
./install.sh
```

`Build dataset` is the first workbench operation. Its popup lets you select
exact source-photo groups from the build spec. V2 deterministically samples a
pinned Hugging Face source, captions each accepted real photo with pinned local
Moondream, freezes the normalized caption, and generates every synthetic
counterpart. Pinned CLIP QA selects the first passing candidate automatically;
failed groups are quarantined with receipts. The builder then applies identical
normalization, embeds analysis images and typed metadata in Parquet, and
validates the package. Byte-identical originals and full lineage remain as
audit sidecars. No paid generation API is configured.

The workbench also initializes specs, validates packages, publishes releases,
and downloads exact groups from a pinned Hugging Face revision. Every field is
explained inline, and the command is shown before it runs.

See [the dataset pipeline guide](docs/dataset-pipeline.md) for the package
layout, complete build-spec schema, all CLI parameters, and release workflow.

Run the offline checks with:

```bash
python3 -m unittest discover -s tests -v
```

The `resnet/` and `moondream/` directories are retained as historical
detector-oriented experiments. They are not the current research protocol.
