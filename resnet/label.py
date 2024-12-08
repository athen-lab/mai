def label_fake(example):
    example["is_synthetic"] = 1.0
    return example


def label_real(example):
    example["is_synthetic"] = 0.0
    return example
