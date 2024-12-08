import math
import torch
import datasets
import transformers
import bitsandbytes
from tqdm import tqdm
from .hyperparams import TEST_SIZE, ANSWER_EOS, IMG_TOKENS, LR, BATCH_SIZE, EPOCHS, GRAD_ACCUM_STEPS

DEVICE = "cuda"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16 # CPU doesn't support float16
MD_REVISION = "2024-07-23"

diffusion_db_dataset = datasets.load_dataset("poloclub/diffusiondb", "2m_random_5k", trust_remote_code=True, split="train")\
                               .select_columns(["image"])\
                               .map(lambda row: {
                                   **row,
                                   "qa": {
                                       "question": "Describe this image.",
                                       "answer": "This is an AI image."
                                   }
                               })\
                               .train_test_split(test_size=TEST_SIZE)

flickr_dataset = datasets.load_dataset("nlphuji/flickr30k", split="test")\
                         .take(5000)\
                         .select_columns(["image"])\
                         .map(lambda row: {
                             **row,
                             "qa": {
                                 "question": "Describe this image.",
                                 "answer": "This is a real image."
                             }
                         })\
                         .train_test_split(test_size=TEST_SIZE)

training_dataset = datasets.concatenate_datasets([diffusion_db_dataset["train"], flickr_dataset["train"]]).shuffle()
test_dataset = datasets.concatenate_datasets([diffusion_db_dataset["test"], flickr_dataset["test"]]).shuffle()

tokenizer = transformers.AutoTokenizer.from_pretrained("vikhyatk/moondream2")
moondream = transformers.AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)

def collate(batch):
    images = []
    all_tokens = []
    all_labels = []

    for sample in batch:
        images.append(sample["image"])

        tokens = [tokenizer.bos_token_id]
        labels = [-100] * (IMG_TOKENS + 1)

        qa = sample["qa"]
        q_t = tokenizer(
            f"\n\nQuestion: {qa['question']}\n\nAnswer:",
            add_special_tokens=False,
        ).input_ids
        tokens.extend(q_t)
        labels.extend([-100] * len(q_t))

        a_t = tokenizer(
            f" {qa['answer']}{ANSWER_EOS}",
            add_special_tokens=False,
        ).input_ids
        tokens.extend(a_t)
        labels.extend(a_t)

        all_tokens.append(tokens)
        all_labels.append(labels)

    longest_label_len = -1
    for label in all_labels:
        longest_label_len = max(longest_label_len, len(label))

    all_attn_masks = []
    for i in range(len(batch)):
        label_len = len(all_labels[i])
        pad_len = longest_label_len - label_len

        all_labels[i].extend([-100] * pad_len)
        all_tokens[i].extend([tokenizer.eos_token_id] * pad_len)
        all_attn_masks.append([1] * label_len + [0] * pad_len)

    return (
        images,
        torch.stack([torch.tensor(token, dtype=torch.long) for token in all_tokens]),
        torch.stack([torch.tensor(label, dtype=torch.long) for label in all_labels]),
        torch.stack([torch.tensor(mask, dtype=torch.bool) for mask in all_attn_masks]),
    )

def compute_loss(batch):
    images, tokens, labels, masks = batch

    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    masks = masks.to(DEVICE)

    with torch.no_grad():
        img_embeds = moondream.vision_encoder(images)

    token_embeds = moondream.text_model.get_input_embeddings()(tokens)

    # start with embedding vector that represents bos, then insert image embeds, then the rest of the token embeds
    # <BOS> + the image + all the tokens
    inputs_embeds = torch.cat((token_embeds[:, 0:1, :], img_embeds, token_embeds[:, 1:, :]), dim=1)

    outputs = moondream.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=masks,
    )

    return outputs.loss

def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

dataloaders = {
    "train": torch.utils.data.DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )
}

moondream.text_model.train()
moondream.text_model.transformer.gradient_checkpointing_enable()

total_steps = EPOCHS * len(dataloaders["train"]) // GRAD_ACCUM_STEPS
optimizer = bitsandbytes.optim.Adam8bit(
    [{"params": moondream.text_model.parameters()}],
    lr=LR*0.1,
    betas=(0.9, 0.95),
    eps=1e-6,
)

i = 0
for epoch in range(EPOCHS):
    for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        i += 1

        loss = compute_loss(batch)
        loss.backward()

        if i % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

moondream.save_pretrained("checkpoints/moondream-mai")
