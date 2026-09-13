
import json
import hashlib
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def checkpoint_iteration(name):
    nums = re.findall(r"(\d+)", name)
    return int(nums[-1]) if nums else None


def build_checkpoint_mapping(samples, out_path=None, seed=42):

    all_ckpts = sorted({
        ckpt
        for sample in samples
        for ckpt in sample.get("responses", {}).keys()
    })

    rng = random.Random(seed)

    anon_names = [
        f"ckpt_{chr(ord('A') + i)}"
        for i in range(len(all_ckpts))
    ]

    rng.shuffle(anon_names)

    mapping = {}

    for ckpt, anon in zip(all_ckpts, anon_names):

        mapping[ckpt] = {
            "anonymous_id": anon,
            "iteration": checkpoint_iteration(ckpt),
        }

    if out_path:
        save_json(mapping, out_path)

    return mapping


def shuffled_candidate_order(
    checkpoint_names,
    sample_id,
    stage,
    seed=42,
):

    h = int(
        hashlib.sha256(
            f"{seed}|{sample_id}|{stage}".encode()
        ).hexdigest()[:8],
        16,
    )

    rng = random.Random(h)

    order = list(checkpoint_names)
    rng.shuffle(order)

    return order


def validate_dataset(samples):

    if not samples:
        raise ValueError("Dataset is empty.")

    checkpoint_sets = [
        set(sample.get("responses", {}).keys())
        for sample in samples
    ]

    common = set.intersection(*checkpoint_sets)

    print("Number of samples:", len(samples))
    print("Checkpoints common to all samples:", len(common))
    print(sorted(common))

    missing_questions = [
        s.get("sample_id")
        for s in samples
        if not s.get("question")
    ]

    missing_images = [
        s.get("sample_id")
        for s in samples
        if not s.get("image_path")
    ]

    if missing_questions:
        print(
            "WARNING:",
            len(missing_questions),
            "samples missing questions"
        )

    if missing_images:
        print(
            "WARNING:",
            len(missing_images),
            "samples missing image paths"
        )

    counts = pd.Series([
        len(s.get("responses", {}))
        for s in samples
    ])

    print("\nResponses per sample:")
    print(counts.describe())
