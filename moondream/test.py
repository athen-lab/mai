import torch
import datasets
import transformers
import pathlib

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

diffusion_db_dataset = datasets.load_dataset("poloclub/diffusiondb", "2m_random_5k", trust_remote_code=True, split="train")\
                               .shuffle()\
                               .take(100)\
                               .select_columns(["image"])\
                               .map(lambda row: {
                                   **row,
                                   "qa": {
                                       "question": "Describe this image.",
                                       "answer": "This is an AI image."
                                   }
                               })

flickr_dataset = datasets.load_dataset("nlphuji/flickr30k", split="test")\
                         .shuffle()\
                         .take(100)\
                         .select_columns(["image"])\
                         .map(lambda row: {
                             **row,
                             "qa": {
                                 "question": "Describe this image.",
                                 "answer": "This is a real image."
                             }
                         })

dataset = datasets.concatenate_datasets([diffusion_db_dataset, flickr_dataset]).shuffle()

pathlib.Path("./samples").mkdir(parents=True, exist_ok=True)

for i, sample in enumerate(dataset):
    sample['image'].save(f"samples/{i}.png", "PNG")

    md_answer = moondream.answer_question(
        moondream.encode_image(sample['image']),
        sample['qa']['question'],
        tokenizer=tokenizer,
        num_beams=4,
        no_repeat_ngram_size=5,
        early_stopping=True
    )

    if i < 3:
        print('Question:', sample['qa']['question'])
        print('Ground Truth:', sample['qa']['answer'])
        print('Moondream:', md_answer)
    else:
        break
