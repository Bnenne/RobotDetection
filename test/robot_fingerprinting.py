"""Stage 7: Identify and fingerprint each robot."""

import torch
from typing import Any
from torch import Tensor
import torch.nn.functional as F
from sklearn.cluster import KMeans


def execute(
    previous_artifacts: dict[str, Any]
) -> dict[str, Any]:
    """Fingerprint robots from video."""
    # TODO: Implement robot fingerprinting


    teams = ["7072", "5125", "3140", "2190", "5005", "8772"]
    teams: list[str]

    portfolios: dict[int, dict] = {}

    for team in teams:
        portfolios[int(team)] = {
            "track_ids": [],
            "bumpers": [],
        }

        # TODO: format data with track_id pointing to it's bumper readings and reid embeddings
    video_data = previous_artifacts.get("video_data")
    video_data: dict[int, dict]

    track_data: dict[int, dict] = {}

    # Sort video-frame data by track_id
    for feed, frame_data in video_data.items():
        frame_data: dict[int, list[dict]]
        for frame, tracks in frame_data.items():
            for track in tracks:
                track: dict[str, Any]
                track_id = track["track_id"]

                # if entry for track_id doesn't exist, create one
                if track_id not in track_data:
                    track_data[track_id] = {
                        "feed": feed,
                        "avg_embedding": None,
                        "norm_embedding": None,
                        "embeddings": [],
                        "frames": [],
                        "bumpers": []
                    }

                # adding the data
                track_data[track_id]["frames"].append(frame)
                track_data[track_id]["embeddings"].append(track["embedding"])
                for bumper in track["bumpers"]:
                    track_data[track_id]["bumpers"].append(bumper)

    # compute average embedding for each track_id
    for track_id, track in track_data.items():
        tensors: list[Tensor] = []

        # convert embeddings back into tensors and store them
        for embedding in track["embeddings"]:
            tensors.append(torch.tensor(embedding))

        # computed average embedding
        avg_embedding = torch.stack(tensors).mean(dim=0)

        # add raw centroid and normalized embedding back to data
        track_data[track_id]["avg_embedding"] = avg_embedding
        track_data[track_id]["norm_embedding"] = F.normalize(avg_embedding, dim=0)

    track_ids = list(track_data.keys())

    embedding_stack = torch.stack([
        track_data[tid]["norm_embedding"]
        for tid in track_ids
    ])

    embedding_stack = F.normalize(embedding_stack, dim=1)

    kmeans = KMeans(n_clusters=6, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(embedding_stack.cpu().numpy())

    clusters = {}
    for tid, label in zip(track_ids, labels):
        clusters.setdefault(label, []).append(tid)

    new_dict = []

    for k, v in sorted(clusters.items(), key=lambda x: int(x[0])):
        new_dict[int(k)] = v

    return { "fingerprints": new_dict }

        # TODO: Compute similarity scores between each track_id

            # TODO: Use cosine similarity for reid similarity

            # TODO: (something about list similarity and visual similarity in order to create bumper similarity score)

        # TODO: Resolve conflicts between track_id "claims"

        # TODO: Create portfolio of 6 robots with full bumper readings and reid embeddings but no identification

        # TODO: Use bumper readings and compare reid embeddings to database of known robots to determine which robot it is

        # TODO: Update database with new embeddings

    return {
        "fingerprints": {
            "R1": {},
            "R2": {},
            "R3": {},
            "B1": {},
            "B2": {},
            "B3": {},
        }
    }
