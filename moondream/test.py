import torch
import datasets
import transformers
import pathlib
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16 # CPU doesn't support float16
MD_REVISION = "2024-07-23"

tokenizer = transformers.AutoTokenizer.from_pretrained("vikhyatk/moondream2")
moondream = transformers.AutoModelForCausalLM.from_pretrained(
    "./checkpoints/moondream-mai",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)

pathlib.Path("./samples").mkdir(parents=True, exist_ok=True)

img = Image.open("samples/Untitled.jpg")
md_answer = moondream.answer_question(
    moondream.encode_image(img),
    "Describe this image.",
    tokenizer=tokenizer,
    num_beams=4,
    no_repeat_ngram_size=5,
    early_stopping=True,
)

print(md_answer)

# correct_predictions = 0
# for i, sample in enumerate(flickr_dataset):
#     if i > 4:
#         break

#     sample["image"].save(f"samples/{i}.png", "PNG")

#     md_answer = moondream.answer_question(
#         moondream.encode_image(sample['image']),
#         sample['qa']['question'],
#         tokenizer=tokenizer,
#         num_beams=4,
#         no_repeat_ngram_size=5,
#         early_stopping=True
#     )

#     print(f"Question: {sample['qa']['question']}")
#     print(f"Ground truth: {sample['qa']['answer']}")
#     print(f"Moondream: {md_answer}")
#     print()

#     if md_answer.lower() == sample['qa']['answer'].lower():
#         correct_predictions += 1

# print(f"Accuracy: {correct_predictions * 100 / 10}%")
