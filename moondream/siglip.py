import transformers
import torch
import datasets
import sklearn

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = transformers.AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
processor = transformers.AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
nn = sklearn.neighbors.NearestNeighbors(metric="euclidean", radius=1.0)

ds = datasets.load_dataset("ehristoforu/midjourney-images", split="train", trust_remote_code=True, streaming=True)\
                               .select_columns(["image"])\
                               .map(lambda row: {
                                   **row,
                                   "qa": {
                                       "question": "Describe this image.",
                                       "answer": "This is an AI image."
                                   }
                               })\
                               .take(500)

with torch.no_grad():
    inputs = processor(images=[row["image"] for row in ds], return_tensors="pt").to(device)
    image_features = model.get_image_features(**inputs).cpu()

nn.fit(image_features)

used_indices = set()
unique_indices = []
for i, row in enumerate(ds):
    if i in used_indices:
        continue

    feature = image_features[i]

    neighbors = nn.radius_neighbors([feature], radius=1.0, return_distance=False)[0]

    unique_indices.append(i)
    used_indices.update(neighbors)

print(len(unique_indices))
