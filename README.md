# Multilayer Authenticity Identifier (MAI)

MAI studies the representation geometry of provenance-verifiable camera
photographs and fully generated photorealistic images. The immediate goal is an
embedding atlas across encoders, layers, generator families, and controlled
transformations—not another production detector.

This Git repository is the dataset **control plane**: it contains the workbench,
build/validation/publishing code, documentation, and tests. Images and release
metadata are the **data plane** and belong in a Hugging Face dataset repository.
The canonical design is checked in at `specs/v1.json`. Local inputs and build
directories live under ignored `.mai-data/`.

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
exact semantic groups from the build spec. Build downloads screened camera
photographs and generates every synthetic counterpart on demand, caches those
inputs, normalizes them, embeds the analysis images and typed metadata in
Parquet shards, and validates the local package. Byte-identical originals and
provenance receipts remain as audit sidecars. Synthetic images are generated
locally with Diffusers—no paid image-generation API is configured.
The same path builds three groups for a smoke test or all 200 groups for the
preliminary run.

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
