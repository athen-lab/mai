import dotenv
dotenv.load_dotenv()

import os
import sys
import torch
import datasets
import diffusers
import dotenv
import transformers
import argparse
from .hyperparams import MOONDREAM_REVISION

print(f"HF_HOME set to {os.getenv('HF_HOME')}")

# DATASET_SIZE = 10000
# ROWS_PER_DS = 1250
BATCH_SIZE = 4
PARQUET_BATCH_SIZE = 200
SKIP_PARQUET_BATCH = 203

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--message")

args = parser.parse_args()
auth_token = os.getenv("HF_ACCESS_TOKEN")
if not auth_token:
    print("huggingface access token not provided! please use the HF_ACCESS_TOKEN env var.")
    sys.exit(1)
else:
    print("huggingface access token loaded!")

tokenizer = transformers.AutoTokenizer.from_pretrained("vikhyatk/moondream2")
moondream = transformers.AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision=MOONDREAM_REVISION,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda"},
)

pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
    token=auth_token,
    device_map="balanced",
)

def collate(batch):
    images = []
    keywords = []

    for sample in batch:
        images.append(sample["image"])
        keywords.append([""])

    return images, keywords

# flickr_dataset = datasets.load_dataset("nlphuji/flickr30k", split="test", streaming=True)\
#                          .select_columns(["image"])\

wiki_art_dataset = datasets.load_dataset("huggan/wikiart", split="train", streaming=True)\
                           .select_columns(["image"])

# anime_dataset_ft = datasets.Features({"image": datasets.Image(decode=True)})
# anime_dataset = datasets.load_dataset("animelover/danbooru2022", "1-full", trust_remote_code=True, split="train", streaming=True, features=anime_dataset_ft)\
#                         .select_columns(["image"])\
#                         .take(ROWS_PER_DS)\
#                         .add_column("question", ["Describe this image in one sentence. Include the word anime in the sentence."] * ROWS_PER_DS)\
#                         .add_column("keywords", [["anime"]] * ROWS_PER_DS)

# coco_dataset = datasets.load_dataset("detection-datasets/coco", split="train", streaming=True)\
#                        .select_columns(["image"])\
#                        .take(ROWS_PER_DS)\
#                        .add_column("question", ["Describe this image in one sentence."] * ROWS_PER_DS)\
#                        .add_column("keywords", [[""]] * ROWS_PER_DS)

# movie_poster_dataset = datasets.load_dataset("skvarre/movie_posters-100k", split="train", streaming=True)\
#                                .select_columns(["image"])\
#                                .take(ROWS_PER_DS)\
#                                .add_column("question", ["Describe this image in one sentence."] * ROWS_PER_DS)\
#                                .add_column("keywords", [[""]] * ROWS_PER_DS)

# cars_dataset = datasets.load_dataset("tanganke/stanford_cars", split="train", streaming=True)\
#                        .select_columns(["image"])\
#                        .take(ROWS_PER_DS)\
#                        .add_column("question", ["Describe this image in one sentence."] * ROWS_PER_DS)\
#                        .add_column("keywords", [[""]] * ROWS_PER_DS)

# website_dataset = datasets.load_dataset("silatus/1k_Website_Screenshots_and_Metadata", split="train", streaming=True)\
#                           .select_columns(["image"])\
#                           .take(ROWS_PER_DS)\
#                           .add_column("question", ["Describe this image in one sentence."] * ROWS_PER_DS)\
#                           .add_column("keywords", [[""]] * ROWS_PER_DS)

# movie_scene_dataset = datasets.load_dataset("unography/movie-scenes-resized-captioned", split="train", streaming=True)\
#                               .select_columns(["image"])\
#                               .take(ROWS_PER_DS)\
#                               .add_column("question", ["Describe this image in one sentence."] * ROWS_PER_DS)\
#                               .add_column("keywords", [[""]] * ROWS_PER_DS)

# ds = datasets.concatenate_datasets([
#     flickr_dataset,
#     wiki_art_dataset,
#     anime_dataset,
#     coco_dataset,
#     movie_poster_dataset,
#     cars_dataset,
#     website_dataset,
#     movie_scene_dataset,
# ]).cast_column("image", datasets.Image(decode=True)).skip(SKIP_PARQUET_BATCH * PARQUET_BATCH_SIZE)

ds = wiki_art_dataset.cast_column("image", datasets.Image(decode=True))

data_loader = torch.utils.data.DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    collate_fn=collate
)

temp_ds = {
    "image": [],
    "keywords": [],
    "caption": [],
    "generated_image": []
}
temp_ds_size = 0

ds_features = datasets.Features({
    "image": datasets.Image(),
    "keywords": datasets.Sequence(datasets.Value(dtype="string")),
    "caption": datasets.Value(dtype="string"),
    "generated_image": datasets.Image(),
})

generator = torch.Generator(device="cpu").manual_seed(12321313)

batch_count = SKIP_PARQUET_BATCH

for batch_index, batch in enumerate(data_loader):
    images, keywords = batch

    prompts = []
    for i, img in enumerate(images):
        caption = moondream.caption(img, length="normal")["caption"]

        add_keywords = len(keywords[i]) > 0 and keywords[i][0] != ""
        for k in keywords[i]:
            if k and k in caption:
                add_keywords = False
                break

        prompt = caption
        if add_keywords:
            prompt = f"{', '.join(keywords[i])}, {caption}"

        prompts.append(prompt)

    gen_imgs = pipe(
        prompts,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=generator,
        max_sequence_length=512,
    ).images

    temp_ds["image"].extend(images)
    temp_ds["caption"].extend(prompts)
    temp_ds["keywords"].extend(keywords)
    temp_ds["generated_image"].extend(gen_imgs)

    temp_ds_size += BATCH_SIZE

    if temp_ds_size == PARQUET_BATCH_SIZE:
        batch_ds = datasets.Dataset.from_dict(temp_ds, features=ds_features)
        batch_ds.to_parquet(
            f"data/batch_{batch_count}.parquet",
        )
        temp_ds_size = 0
        temp_ds["image"].clear()
        temp_ds["caption"].clear()
        temp_ds["keywords"].clear()
        temp_ds["generated_image"].clear()

        batch_count += 1
