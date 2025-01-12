import os
import torch
import datasets
import diffusers
from .hyperparams import MOONDREAM_REVISION

auth_token = os.getenv("HF_ACCESS_TOKEN")

tokenizer = transformers.AutoTokenizer.from_pretrained("vikhyatk/moondream2")
moondream = transformers.AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision=MOONDREAM_REVISION,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
).to("cuda")

def collate(batch):
    images = []
    questions = []

    for sample in batch:
        images.append(sample["image"])
        questions.append("Describe this image.")

    return images, questions

flickr_dataset = datasets.load_dataset("nlphuji/flickr30k", split="test", streaming=True)\
                         .select_columns(["image"])\
                         .take(1)

wiki_art_dataset = datasets.load_dataset("huggan/wikiart", split="train", streaming=True)\
                           .select_columns(["image"])\
                           .take(1)

anime_dataset = datasets.load_dataset("animelover/danbooru2022", "1-full", trust_remote_code=True, split="train", streaming=True)\
                        .select_columns(["image"])\
                        .take(1)

coco_dataset = datasets.load_dataset("detection-datasets/coco", split="train", streaming=True)\
                       .select_columns(["image"])\
                       .take(1)

movie_poster_dataset = datasets.load_dataset("skvarre/movie_posters-100k", split="train", streaming=True)\
                               .select_columns(["image"])\
                               .take(1)

cars_dataset = datasets.load_dataset("tanganke/stanford_cars", split="train", streaming=True)\
                       .select_columns(["image"])\
                       .take(1)

website_dataset = datasets.load_dataset("silatus/1k_Website_Screenshots_and_Metadata", split="train", streaming=True)\
                          .select_columns(["image"])\
                          .take(1)

movie_scene_dataset = datasets.load_dataset("unography/movie-scenes-resized-captioned", split="train", streaming=True)\
                              .select_columns(["image"])\
                              .take(1)

ds = datasets.concatenate_datasets([
    flickr_dataset,
    wiki_art_dataset,
    anime_dataset,
    coco_dataset,
    movie_poster_dataset,
    cars_dataset,
    website_dataset,
    movie_scene_dataset,
])

data_loader = torch.utils.data.DataLoader(
    ds,
    batch_size=8,
    collate_fn=collate
)

captions = []
for batch in data_loader:
    images, questions = batch
    answers = moondream.batch_answer(
        images=images,
        prompts=questions,
        tokenizer=tokenizer
    )

    for ans in answers:
        print(ans)
        print()

    captions.extend(answers)

ds = ds.add_column("caption", captions)

del moondream

pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
    token=auth_token,
).to("cuda")

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("capybara.png")
