import datasets
import torch
import torch.nn
import torch.optim
import torch.utils.data
from pathlib import Path
from datetime import datetime
from datasets import concatenate_datasets, load_dataset
from label import label_fake, label_real
from model import mai
from augmentation import transform_training, transform_validation
from hyperparams import BATCH_SIZE, EPOCHS
from resnet import MaiRes

TEST_SIZE = 0.1

datasets.logging.set_verbosity(datasets.logging.INFO)


def collate(batch):
    pixel_values = []
    is_synthetic = []
    for row in batch:
        is_synthetic.append(row["is_synthetic"])
        pixel_values.append(torch.tensor(row["pixel_values"]))

    pixel_values = torch.stack(pixel_values, dim=0)
    is_synthetic = torch.tensor(is_synthetic, dtype=torch.float)

    return pixel_values, is_synthetic


def load_data():
    print("loading datasets...")

    diffusion_db_dataset = load_dataset(
        "poloclub/diffusiondb",
        "2m_random_50k",
        trust_remote_code=True,
        split="train",
    )
    flickr_dataset = load_dataset("nlphuji/flickr30k", split="test")
    painting_dataset = load_dataset(
        "keremberke/painting-style-classification", name="full", split="train"
    )
    anime_scene_datatset = load_dataset(
        "animelover/scenery-images", "0-sfw", split="train"
    )
    movie_poaster_dataset = load_dataset("nanxstats/movie-poster-5k", split="train")
    metal_album_art_dataset = load_dataset(
        "Alphonsce/metal_album_covers", split="train[:50%]"
    )

    diffusion_db_dataset = diffusion_db_dataset.select_columns("image")
    diffusion_db_dataset = diffusion_db_dataset.map(label_fake)

    flickr_dataset = flickr_dataset.select_columns("image")
    flickr_dataset = flickr_dataset.map(label_real)

    painting_dataset = painting_dataset.select_columns("image")
    painting_dataset = painting_dataset.map(label_real)
    anime_scene_datatset = anime_scene_datatset.select_columns("image")
    anime_scene_datatset = anime_scene_datatset.map(label_real)

    movie_poaster_dataset = movie_poaster_dataset.select_columns("image")
    movie_poaster_dataset = movie_poaster_dataset.map(label_real)

    metal_album_art_dataset = metal_album_art_dataset.select_columns("image")
    metal_album_art_dataset = metal_album_art_dataset.map(label_real)

    diffusion_split = diffusion_db_dataset.train_test_split(test_size=TEST_SIZE)
    flickr_split = flickr_dataset.train_test_split(test_size=TEST_SIZE)
    painting_split = painting_dataset.train_test_split(test_size=TEST_SIZE)
    anime_scene_split = anime_scene_datatset.train_test_split(test_size=TEST_SIZE)
    movie_poaster_split = movie_poaster_dataset.train_test_split(test_size=TEST_SIZE)
    metal_album_art_split = metal_album_art_dataset.train_test_split(
        test_size=TEST_SIZE
    )

    training_ds = concatenate_datasets(
        [
            diffusion_split["train"],
            flickr_split["train"],
            painting_split["train"],
            anime_scene_split["train"],
            movie_poaster_split["train"],
            metal_album_art_split["train"],
        ]
    )
    validation_ds = concatenate_datasets(
        [
            diffusion_split["test"],
            flickr_split["test"],
            painting_split["test"],
            anime_scene_split["test"],
            movie_poaster_split["test"],
            metal_album_art_split["test"],
        ]
    )

    training_ds = training_ds.map(
        transform_training, remove_columns=["image"], batched=True
    )
    validation_ds = validation_ds.map(
        transform_validation, remove_columns=["image"], batched=True
    )

    training_loader = torch.utils.data.DataLoader(
        training_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        collate_fn=collate,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        collate_fn=collate,
    )

    return training_loader, validation_loader, len(training_ds)


def train():
    training_loader, validation_loader, sample_size = load_data()

    print(f"sample size: {sample_size}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = MaiRes().cuda()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    best_vloss = 1_000_000.0

    Path("models").mkdir(parents=True, exist_ok=True)
    print("models directory created")

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch + 1}:")

        model.train(True)

        running_loss = 0.0
        last_loss = 0.0
        correct = 0.0
        total = 0.0
        accuracy = 0.0
        i = 0

        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs = inputs.cuda()

            labels = labels.cuda()
            labels = labels.view(-1, 1)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()  # Applying a threshold of 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / sample_size

            running_loss += loss.item()
            if i % 10 == 9:
                last_loss = running_loss / 10  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                running_loss = 0.0

        print(f"ACCURACY {accuracy}")

        # validation step
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, validation_data in enumerate(validation_loader):
                vinputs, vlabels = validation_data
                vinputs = vinputs.cuda()
                vlabels = vlabels.cuda()
                vlabels = vlabels.view(-1, 1)

                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)

                running_vloss += vloss

        avg_validation_loss = running_vloss / (i + 1)
        print("LOSS train {} valid {}".format(last_loss, avg_validation_loss))

        if avg_validation_loss < best_vloss:
            best_vloss = avg_validation_loss
            model_path = f"model/mai_{timestamp}_{epoch}"
            torch.save(model.state_dict(), model_path)
            print("current model state saved")


train()
