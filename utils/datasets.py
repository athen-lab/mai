import datasets

def split_streaming_dataset(ds: datasets.IterableDataset, total_size: int, test_size: float) -> dict[str, datasets.IterableDataset]:
    size = round(total_size * (1 - test_size))
    return {
        "train": ds.take(size),
        "test": ds.skip(size).take(total_size - size),
    }
