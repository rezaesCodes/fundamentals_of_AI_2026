"""Generate deterministic datasets for the integrated engineering mini-project.

Run from the project folder:
    python generate_project_data.py --seed 12345678 --output notebooks/data

The generator writes the time-series, feeder-profile, and synthetic speed-limit-sign
datasets used by the student assignment. The image-generation code is intentionally
kept outside the student notebook.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


def make_load(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_hours = 24 * 90
    timestamps = pd.date_range("2025-01-01", periods=n_hours, freq="h")
    t = np.arange(n_hours)
    hour = timestamps.hour.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    daily = 42 * np.sin(2 * np.pi * (hour - 8) / 24)
    second = 11 * np.sin(4 * np.pi * (hour - 16) / 24)
    weekly = np.where(dow < 5, 16, -12)
    trend = 0.018 * t
    temperature = 19 + 8 * np.sin(2 * np.pi * t / (24 * 30)) + 3 * np.sin(2 * np.pi * (hour - 14) / 24)
    feeder_load = 210 + daily + second + weekly + 0.8 * (temperature - 19) + trend
    feeder_load += rng.normal(0, 5.0, n_hours)
    temperature += rng.normal(0, 1.0, n_hours)
    df = pd.DataFrame({"timestamp": timestamps, "load_kw": feeder_load, "temperature_c": temperature})
    for col, rate in [("load_kw", 0.022), ("temperature_c", 0.018)]:
        ix = rng.choice(df.index, int(rate * len(df)), replace=False)
        df.loc[ix, col] = np.nan
    ix = rng.choice(df.index, int(0.012 * len(df)), replace=False)
    df.loc[ix, "load_kw"] = rng.uniform(550, 800, len(ix))
    ix = rng.choice(df.index, int(0.006 * len(df)), replace=False)
    df.loc[ix, "temperature_c"] = rng.choice([-40, 85], len(ix))
    ix = rng.choice(df.index, int(0.006 * len(df)), replace=False)
    df = df.drop(index=ix)
    df = pd.concat([df, df.sample(frac=0.008, random_state=seed)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_profiles(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 101)
    rows = []
    profile_types = ["Base-load", "Peaky", "Variable", "Night-heavy"]
    means = {"Base-load": 180, "Peaky": 145, "Variable": 160, "Night-heavy": 155}
    variability_levels = {"Base-load": 0.08, "Peaky": 0.32, "Variable": 0.24, "Night-heavy": 0.16}
    peak_levels = {"Base-load": 1.22, "Peaky": 1.85, "Variable": 1.55, "Night-heavy": 1.35}
    night_levels = {"Base-load": 0.38, "Peaky": 0.22, "Variable": 0.31, "Night-heavy": 0.62}
    for feeder_id in range(240):
        kind = profile_types[feeder_id % len(profile_types)]
        mean = means[kind] + rng.normal(0, 12)
        variability = variability_levels[kind] + rng.normal(0, 0.025)
        peak_factor = peak_levels[kind] + rng.normal(0, 0.08)
        night_ratio = night_levels[kind] + rng.normal(0, 0.035)
        load_factor = np.clip(1 / peak_factor + rng.normal(0, 0.025), 0.25, 0.95)
        rows.append({
            "feeder_id": f"F{feeder_id + 1:03d}",
            "mean_load_kw": mean,
            "std_load_kw": max(1, mean * variability),
            "peak_load_kw": mean * peak_factor,
            "load_factor": load_factor,
            "night_energy_ratio": np.clip(night_ratio, 0.05, 0.9),
            "peak_to_mean": peak_factor,
            "variability_index": variability,
            "reference_profile": kind,
        })
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in ["feeder_id", "reference_profile"]]
    for col in feature_cols:
        ix = rng.choice(df.index, 5, replace=False)
        df.loc[ix, col] = np.nan
    anomalies = [
        ("mean_load_kw", 950), ("std_load_kw", 600), ("peak_load_kw", 1800),
        ("load_factor", 1.8), ("night_energy_ratio", -0.4),
        ("peak_to_mean", 9.0), ("variability_index", 3.5),
    ]
    for col, value in anomalies:
        ix = rng.choice(df.index, 2, replace=False)
        df.loc[ix, col] = value
    df = pd.concat([df, df.sample(n=4, random_state=seed)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def find_bold_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_speed_sign(number="50", alteration=None):
    size = 96
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    margin = 7
    draw.ellipse(
        (margin, margin, size - margin, size - margin),
        fill=(255, 255, 255),
        outline=(210, 20, 30),
        width=8,
    )
    font = find_bold_font(40 if len(number) == 2 else 31)
    bbox = draw.textbbox((0, 0), number, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) / 2 - bbox[0]
    y = (size - text_h) / 2 - bbox[1] - 2
    draw.text((int(x), int(y)), number, font=font, fill=(5, 5, 5))

    if alteration == "slash":
        draw.line((10, 12, 86, 84), fill=(0, 0, 0), width=8)
    elif alteration == "sticker":
        draw.rounded_rectangle((42, 27, 72, 66), radius=3, fill=(230, 230, 230), outline=(65, 65, 65), width=2)
    elif alteration == "missing_digit":
        draw.rectangle((40, 29, 73, 65), fill=(255, 255, 255))
    elif alteration == "extra_bar":
        draw.line((10, 46, 86, 46), fill=(0, 0, 0), width=7)
    return image


ALTERATIONS = ("changed_to_11", "changed_to_88", "slash", "sticker", "missing_digit", "extra_bar")


def render_sign_pixels(number, alteration, rng, normal=True):
    image = draw_speed_sign(number, alteration)
    if normal:
        brightness = float(rng.uniform(0.97, 1.03))
        noise_sigma = 1.0
    else:
        brightness = float(rng.uniform(0.97, 1.03))
        noise_sigma = 1.5
    image = ImageEnhance.Brightness(image).enhance(brightness)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    gray += rng.normal(0.0, noise_sigma, size=gray.shape)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    small = Image.fromarray(gray).resize((32, 32), Image.Resampling.LANCZOS)
    return np.asarray(small, dtype=np.uint8)


def make_traffic_signs(seed: int) -> pd.DataFrame:
    """Create 100 normal training images and a balanced 900-image test set."""
    rng = np.random.default_rng(seed + 303)
    rows = []
    pixel_names = [f"pixel_{r:02d}_{c:02d}" for r in range(32) for c in range(32)]

    def add_image(split, label, alteration):
        number = "11" if alteration == "changed_to_11" else "88" if alteration == "changed_to_88" else "50"
        image = render_sign_pixels(number, None if alteration in ("normal", "changed_to_11", "changed_to_88") else alteration,
                                   rng, normal=(label == 0))
        row = {
            "split": split,
            "label": int(label),
            "alteration": alteration,
        }
        row.update({name: int(value) for name, value in zip(pixel_names, image.reshape(-1))})
        rows.append(row)

    for _ in range(100):
        add_image("train", 0, "normal")
    for _ in range(450):
        add_image("test", 0, "normal")

    suspicious_rows = []
    for i in range(450):
        alteration = ALTERATIONS[i % len(ALTERATIONS)]
        number = "11" if alteration == "changed_to_11" else "88" if alteration == "changed_to_88" else "50"
        image = render_sign_pixels(number, None if alteration in ("changed_to_11", "changed_to_88") else alteration,
                                   rng, normal=False)
        suspicious_rows.append((alteration, image))
    order = rng.permutation(len(suspicious_rows))
    for idx in order:
        alteration, image = suspicious_rows[int(idx)]
        row = {"split": "test", "label": 1, "alteration": alteration}
        row.update({name: int(value) for name, value in zip(pixel_names, image.reshape(-1))})
        rows.append(row)

    df = pd.DataFrame(rows)
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].sample(frac=1, random_state=seed).copy()
    result = pd.concat([train, test], ignore_index=True)
    result.insert(0, "image_id", [f"S{i + 1:04d}" for i in range(len(result))])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345678)
    parser.add_argument("--output", type=Path, default=Path("notebooks/data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    load_df = make_load(args.seed)
    profile_df = make_profiles(args.seed)
    sign_df = make_traffic_signs(args.seed)
    load_df.to_csv(args.output / "electrical_load_timeseries.csv", index=False)
    profile_df.to_csv(args.output / "electrical_profiles.csv", index=False)
    sign_df.to_csv(args.output / "traffic_sign_dataset.csv", index=False)
    stale_symbols = args.output / "hopfield_patterns.csv"
    if stale_symbols.exists():
        stale_symbols.unlink()

    print(f"Generated three datasets with seed {args.seed} in {args.output.resolve()}")
    print("Time-series rows:", len(load_df))
    print("Feeder-profile rows:", len(profile_df))
    print("Traffic-sign rows:", len(sign_df))
    print("Traffic training rows:", int((sign_df["split"] == "train").sum()))
    print("Traffic testing rows:", int((sign_df["split"] == "test").sum()))


if __name__ == "__main__":
    main()
