#!/usr/bin/env python3
"""
Deep Steganography — Robustness Evaluation Script
Tests how well the hidden secret survives various image attacks.

Usage:
    python evaluate.py [--weights PATH] [--data_dir DIR] [--output_dir DIR]
"""

import argparse
import io
import os
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

from keras.layers import Input, Conv2D, concatenate, GaussianNoise
from keras.models import Model
from keras.preprocessing import image
import keras.ops as K


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Deep Steganography robustness")
    p.add_argument("--weights", type=str, default="./output/model_weights.weights.h5")
    p.add_argument("--data_dir", type=str, default="./tiny-imagenet-200")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--train_per_class", type=int, default=10)
    p.add_argument("--test_images", type=int, default=500)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset (same as train.py)
# ---------------------------------------------------------------------------
def load_dataset_small(data_dir, num_images_per_class_train=10,
                       num_images_test=500):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    X_train, X_test = [], []

    for c in os.listdir(train_dir):
        c_dir = os.path.join(train_dir, c, "images")
        if not os.path.isdir(c_dir):
            continue
        c_imgs = os.listdir(c_dir)
        random.shuffle(c_imgs)
        for img_name in c_imgs[:num_images_per_class_train]:
            img = image.load_img(os.path.join(c_dir, img_name))
            X_train.append(image.img_to_array(img))
    random.shuffle(X_train)

    test_img_dir = os.path.join(test_dir, "images")
    test_imgs = os.listdir(test_img_dir)
    random.shuffle(test_imgs)
    for img_name in test_imgs[:num_images_test]:
        img = image.load_img(os.path.join(test_img_dir, img_name))
        X_test.append(image.img_to_array(img))

    return np.array(X_train), np.array(X_test)


# ---------------------------------------------------------------------------
# Loss functions (needed for model compilation)
# ---------------------------------------------------------------------------
BETA = 1.0

def rev_loss(s_true, s_pred):
    return BETA * K.sum(K.square(s_true - s_pred))

def full_loss(y_true, y_pred):
    s_true, c_true = y_true[..., 0:3], y_true[..., 3:6]
    s_pred, c_pred = y_pred[..., 0:3], y_pred[..., 3:6]
    s_loss = rev_loss(s_true, s_pred)
    c_loss = K.sum(K.square(c_true - c_pred))
    return s_loss + c_loss


# ---------------------------------------------------------------------------
# Model architecture (same as train.py)
# ---------------------------------------------------------------------------
def make_encoder(input_size):
    input_S = Input(shape=input_size)
    input_C = Input(shape=input_size)

    x3 = Conv2D(50, (3, 3), padding="same", activation="relu", name="conv_prep0_3x3")(input_S)
    x4 = Conv2D(10, (4, 4), padding="same", activation="relu", name="conv_prep0_4x4")(input_S)
    x5 = Conv2D(5,  (5, 5), padding="same", activation="relu", name="conv_prep0_5x5")(input_S)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(50, (3, 3), padding="same", activation="relu", name="conv_prep1_3x3")(x)
    x4 = Conv2D(10, (4, 4), padding="same", activation="relu", name="conv_prep1_4x4")(x)
    x5 = Conv2D(5,  (5, 5), padding="same", activation="relu", name="conv_prep1_5x5")(x)
    x = concatenate([x3, x4, x5])

    x = concatenate([input_C, x])

    for i in range(5):
        suffix = str(i)
        x3 = Conv2D(50, (3, 3), padding="same", activation="relu",
                     name=f"conv_hid{suffix}_3x3")(x)
        x4 = Conv2D(10, (4, 4), padding="same", activation="relu",
                     name=f"conv_hid{suffix}_4x4")(x)
        x5 = Conv2D(5,  (5, 5), padding="same", activation="relu",
                     name=f"conv_hid{suffix}_5x5")(x)
        x = concatenate([x3, x4, x5])

    output_Cprime = Conv2D(3, (3, 3), padding="same", activation="relu",
                           name="output_C")(x)
    return Model(inputs=[input_S, input_C], outputs=output_Cprime, name="Encoder")


def make_decoder(input_size):
    reveal_input = Input(shape=input_size)
    input_with_noise = GaussianNoise(0.01, name="output_C_noise")(reveal_input)

    x = input_with_noise
    for i in range(5):
        suffix = str(i)
        inp = x if i > 0 else input_with_noise
        x3 = Conv2D(50, (3, 3), padding="same", activation="relu",
                     name=f"conv_rev{suffix}_3x3")(inp if i == 0 else x)
        x4 = Conv2D(10, (4, 4), padding="same", activation="relu",
                     name=f"conv_rev{suffix}_4x4")(inp if i == 0 else x)
        x5 = Conv2D(5,  (5, 5), padding="same", activation="relu",
                     name=f"conv_rev{suffix}_5x5")(inp if i == 0 else x)
        x = concatenate([x3, x4, x5])

    output_Sprime = Conv2D(3, (3, 3), padding="same", activation="tanh",
                           name="output_S")(x)
    return Model(inputs=reveal_input, outputs=output_Sprime, name="Decoder")


def make_model(input_size):
    input_S = Input(shape=input_size)
    input_C = Input(shape=input_size)

    encoder = make_encoder(input_size)
    decoder = make_decoder(input_size)
    decoder.compile(optimizer="adam", loss=rev_loss)

    output_Cprime = encoder([input_S, input_C])

    decoder_fixed = make_decoder(input_size)
    decoder_fixed._name = "DecoderFixed"
    for src, dst in zip(decoder.layers, decoder_fixed.layers):
        dst.set_weights(src.get_weights())
    decoder_fixed.trainable = False

    output_Sprime = decoder_fixed(output_Cprime)

    autoencoder = Model(inputs=[input_S, input_C],
                        outputs=concatenate([output_Sprime, output_Cprime]))
    autoencoder.compile(optimizer="adam", loss=full_loss)

    return encoder, decoder, autoencoder, decoder_fixed


# ---------------------------------------------------------------------------
# Attack functions
# ---------------------------------------------------------------------------
def attack_jpeg(images, quality=50):
    attacked = np.zeros_like(images)
    for i in range(len(images)):
        img_uint8 = (np.clip(images[i], 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        attacked[i] = np.array(Image.open(buf)).astype(np.float32) / 255.0
    return attacked


def attack_gaussian_noise(images, sigma=0.03):
    noise = np.random.normal(0, sigma, images.shape).astype(np.float32)
    return np.clip(images + noise, 0, 1)


def attack_gaussian_blur(images, radius=1.0):
    attacked = np.zeros_like(images)
    for i in range(len(images)):
        img_uint8 = (np.clip(images[i], 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        attacked[i] = np.array(blurred).astype(np.float32) / 255.0
    return attacked


def attack_crop_resize(images, crop_fraction=0.8):
    attacked = np.zeros_like(images)
    h, w = images.shape[1], images.shape[2]
    ch, cw = int(h * crop_fraction), int(w * crop_fraction)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    for i in range(len(images)):
        img_uint8 = (np.clip(images[i], 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        cropped = pil_img.crop((x0, y0, x0 + cw, y0 + ch))
        resized = cropped.resize((w, h), Image.BILINEAR)
        attacked[i] = np.array(resized).astype(np.float32) / 255.0
    return attacked


def attack_brightness(images, factor=1.3):
    return np.clip(images * factor, 0, 1)


def pixel_error(original, reconstructed):
    return np.sqrt(np.mean(np.square(255 * (original - reconstructed))))


def chi_square_lsb_score(img_uint8, min_expected=5):
    """
    Chi-square LSB steganalysis (StegExpose-style).
    Low score = LSB-like (pairs equalized). High score = natural image.
    Returns mean chi2 per channel (averaged over RGB).
    """
    scores = []
    for ch in range(3):
        counts = np.bincount(img_uint8[..., ch].flatten(), minlength=256)
        even_counts = counts[0::2]  # 0, 2, 4, ...
        odd_counts = counts[1::2]   # 1, 3, 5, ...
        total = even_counts + odd_counts
        mask = total >= min_expected
        # Chi2 per pair: (a-b)^2 / (a+b), sum over pairs
        diff_sq = (even_counts - odd_counts).astype(np.float64) ** 2
        chi2_vals = np.where(mask, diff_sq / np.maximum(total.astype(np.float64), 1), 0)
        scores.append(np.sum(chi2_vals))
    return np.mean(scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    X_train_orig, _ = load_dataset_small(
        args.data_dir, args.train_per_class, args.test_images
    )
    X_train = X_train_orig / 255.0
    input_S = X_train[: X_train.shape[0] // 2]
    input_C = X_train[X_train.shape[0] // 2:]
    print(f"Secret images: {input_S.shape[0]}, Cover images: {input_C.shape[0]}")

    print("Building model and loading weights...")
    _, reveal_model, autoencoder_model, _ = make_model(input_S.shape[1:])
    autoencoder_model.load_weights(args.weights)

    # Extract the reveal (decoder) from the autoencoder
    # DecoderFixed (new) or Decoder (older saved models)
    try:
        reveal_from_ae = autoencoder_model.get_layer("DecoderFixed")
    except ValueError:
        reveal_from_ae = autoencoder_model.get_layer("Decoder")

    print("Generating container (stego) images...")
    decoded = autoencoder_model.predict([input_S, input_C], verbose=0)
    decoded_S, decoded_C = decoded[..., 0:3], decoded[..., 3:6]
    container_images = decoded_C

    # Baseline metrics
    s_err_base = pixel_error(input_S, decoded_S)
    c_err_base = pixel_error(input_C, decoded_C)
    print(f"\nBaseline — S error: {s_err_base:.2f}, C error: {c_err_base:.2f}")

    # Define attacks
    attacks = {
        "No Attack":      lambda img: img,
        "JPEG Q=75":      lambda img: attack_jpeg(img, quality=75),
        "JPEG Q=50":      lambda img: attack_jpeg(img, quality=50),
        "JPEG Q=25":      lambda img: attack_jpeg(img, quality=25),
        "JPEG Q=10":      lambda img: attack_jpeg(img, quality=10),
        "Noise σ=0.01":   lambda img: attack_gaussian_noise(img, sigma=0.01),
        "Noise σ=0.03":   lambda img: attack_gaussian_noise(img, sigma=0.03),
        "Noise σ=0.10":   lambda img: attack_gaussian_noise(img, sigma=0.10),
        "Blur r=0.5":     lambda img: attack_gaussian_blur(img, radius=0.5),
        "Blur r=1.0":     lambda img: attack_gaussian_blur(img, radius=1.0),
        "Blur r=2.0":     lambda img: attack_gaussian_blur(img, radius=2.0),
        "Crop 90%":       lambda img: attack_crop_resize(img, crop_fraction=0.9),
        "Crop 80%":       lambda img: attack_crop_resize(img, crop_fraction=0.8),
        "Crop 70%":       lambda img: attack_crop_resize(img, crop_fraction=0.7),
        "Bright x1.3":    lambda img: attack_brightness(img, factor=1.3),
        "Bright x0.7":    lambda img: attack_brightness(img, factor=0.7),
    }

    # Run attacks
    attack_results = {}

    print(f"\n{'Attack':<18} | {'Secret RMSE':>12} | {'Container RMSE':>14}")
    print("-" * 52)

    for name, attack_fn in attacks.items():
        attacked_C = attack_fn(container_images)
        recovered_S = reveal_from_ae.predict(attacked_C, verbose=0)

        s_err = pixel_error(input_S, recovered_S)
        c_err = pixel_error(input_C, attacked_C)

        attack_results[name] = {
            "attacked_C": attacked_C,
            "recovered_S": recovered_S,
            "s_error": s_err,
            "c_error": c_err,
        }
        print(f"{name:<18} | {s_err:>12.2f} | {c_err:>14.2f}")

    print("\nSecret RMSE = how well the secret is recovered (lower = better)")
    print("Container RMSE = how much the attack distorted the cover (lower = milder)")

    # --- Bar chart ---
    names = list(attack_results.keys())
    s_errors = [attack_results[n]["s_error"] for n in names]

    colors = ["#2ecc71"] + ["#e74c3c"] * (len(names) - 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(names)), s_errors, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Secret Recovery RMSE (per pixel, 0-255 scale)")
    ax.set_title("Steganography Robustness: Secret Recovery Error Under Various Attacks")

    for bar, val in zip(bars, s_errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(y=s_errors[0], color="#2ecc71", linestyle="--", alpha=0.5,
               label="Baseline (no attack)")
    ax.legend()
    plt.tight_layout()
    chart_path = os.path.join(args.output_dir, "robustness_chart.png")
    plt.savefig(chart_path, dpi=150)
    print(f"\nBar chart saved to {chart_path}")

    # --- Visual samples ---
    sample_attacks = ["No Attack", "JPEG Q=25", "Noise σ=0.10",
                      "Blur r=2.0", "Crop 70%", "Bright x0.7"]
    n_samples = 4
    n_attacks = len(sample_attacks)
    sample_indices = [random.randint(0, len(input_S) - 1) for _ in range(n_samples)]

    fig, axes = plt.subplots(n_attacks, n_samples * 3,
                             figsize=(n_samples * 9, n_attacks * 3))

    for row, attack_name in enumerate(sample_attacks):
        res = attack_results[attack_name]
        for col, idx in enumerate(sample_indices):
            base_col = col * 3

            ax = axes[row, base_col]
            ax.imshow(np.clip(res["attacked_C"][idx], 0, 1))
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Container #{col+1}", fontsize=9)
            if col == 0:
                ax.set_ylabel(attack_name, fontsize=9, rotation=90, labelpad=40)

            ax = axes[row, base_col + 1]
            ax.imshow(np.clip(res["recovered_S"][idx], 0, 1))
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Recovered #{col+1}", fontsize=9)

            ax = axes[row, base_col + 2]
            ax.imshow(np.clip(input_S[idx], 0, 1))
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Original Secret #{col+1}", fontsize=9)

    plt.tight_layout()
    vis_path = os.path.join(args.output_dir, "robustness_samples.png")
    plt.savefig(vis_path, dpi=150)
    print(f"Visual samples saved to {vis_path}")

    # --- Error histograms (Paper Figure 4, right) ---
    diff_S_flat = (255 * np.abs(decoded_S - input_S)).flatten()
    diff_C_flat = (255 * np.abs(decoded_C - input_C)).flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(diff_C_flat, bins=100, density=True, alpha=0.75, color="#3498db")
    ax1.set_title("Distribution of Error in Cover Image")
    ax1.set_xlabel("Pixel error (0-255)")
    ax1.set_ylabel("Density")
    ax1.set_xlim(0, 80)

    ax2.hist(diff_S_flat, bins=100, density=True, alpha=0.75, color="#e74c3c")
    ax2.set_title("Distribution of Error in Secret Image")
    ax2.set_xlabel("Pixel error (0-255)")
    ax2.set_ylabel("Density")
    ax2.set_xlim(0, 80)

    plt.tight_layout()
    hist_path = os.path.join(args.output_dir, "error_histograms.png")
    plt.savefig(hist_path, dpi=150)
    print(f"Error histograms saved to {hist_path}")

    # --- Residual analysis (Paper Section 3.1, Figure 7) ---
    # If the original cover is leaked, what can be seen in the residual?
    n_res = 4
    enhancements = [5, 10, 20]
    res_indices = [random.randint(0, len(input_S) - 1) for _ in range(n_res)]

    n_cols = 2 + len(enhancements)  # cover, container, residual@5x, 10x, 20x
    fig, axes = plt.subplots(n_res, n_cols, figsize=(n_cols * 3, n_res * 3))

    for row, idx in enumerate(res_indices):
        # Original cover
        ax = axes[row, 0]
        ax.imshow(np.clip(input_C[idx], 0, 1))
        ax.axis("off")
        if row == 0:
            ax.set_title("Original Cover", fontsize=10)

        # Container (stego) image
        ax = axes[row, 1]
        ax.imshow(np.clip(decoded_C[idx], 0, 1))
        ax.axis("off")
        if row == 0:
            ax.set_title("Container", fontsize=10)

        # Residual at different enhancements
        residual = np.abs(decoded_C[idx] - input_C[idx])
        for j, enhance in enumerate(enhancements):
            ax = axes[row, 2 + j]
            ax.imshow(np.clip(residual * enhance, 0, 1))
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Residual ×{enhance}", fontsize=10)

    plt.suptitle("Residual Analysis: What Is Visible If the Original Cover Is Leaked?",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    res_path = os.path.join(args.output_dir, "residual_analysis.png")
    plt.savefig(res_path, dpi=150, bbox_inches="tight")
    print(f"Residual analysis saved to {res_path}")

    # --- Bit sensitivity analysis (Paper Section 4, Figure 9) ---
    # Flip each bit position (0-7) in each color channel of the container
    # and measure the effect on the recovered secret across all channels.
    print("\nRunning bit sensitivity analysis...")
    container_uint8 = (np.clip(container_images, 0, 1) * 255).astype(np.uint8)
    channels = ["Red", "Green", "Blue"]
    bit_positions = list(range(8))

    # Baseline secret from unmodified container
    baseline_S = reveal_from_ae.predict(container_images, verbose=0)

    # impact_on_container[src_ch][bit] = [effect_R, effect_G, effect_B]
    # impact_on_secret[src_ch][bit] = [effect_R, effect_G, effect_B]
    impact_on_container = np.zeros((3, 8, 3))
    impact_on_secret = np.zeros((3, 8, 3))

    for ch in range(3):
        for bit in bit_positions:
            flipped = container_uint8.copy()
            flipped[..., ch] = flipped[..., ch] ^ (1 << bit)
            flipped_float = flipped.astype(np.float32) / 255.0

            # Effect on container itself (control)
            for out_ch in range(3):
                impact_on_container[ch, bit, out_ch] = np.mean(
                    np.abs(flipped_float[..., out_ch] - container_images[..., out_ch]) * 255
                )

            # Effect on recovered secret
            recovered_flipped = reveal_from_ae.predict(flipped_float, verbose=0)
            for out_ch in range(3):
                impact_on_secret[ch, bit, out_ch] = np.mean(
                    np.abs(recovered_flipped[..., out_ch] - baseline_S[..., out_ch]) * 255
                )

        print(f"  {channels[ch]} channel done")

    # Plot: 2 rows x 3 cols (top=container control, bottom=secret impact)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    bar_width = 0.25
    x = np.arange(8)
    colors_rgb = ["#e74c3c", "#2ecc71", "#3498db"]

    for col, src_ch in enumerate(range(3)):
        # Top row: effect on container (control)
        ax = axes[0, col]
        for out_ch in range(3):
            ax.bar(x + out_ch * bar_width, impact_on_container[src_ch, :, out_ch],
                   bar_width, label=f"{channels[out_ch]}", color=colors_rgb[out_ch], alpha=0.8)
        ax.set_title(f"Flip {channels[src_ch]} bit → Container", fontsize=10)
        ax.set_xlabel("Bit position")
        ax.set_ylabel("Mean pixel change")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([str(b) for b in bit_positions])
        if col == 0:
            ax.legend(fontsize=8)

        # Bottom row: effect on secret
        ax = axes[1, col]
        for out_ch in range(3):
            ax.bar(x + out_ch * bar_width, impact_on_secret[src_ch, :, out_ch],
                   bar_width, label=f"{channels[out_ch]}", color=colors_rgb[out_ch], alpha=0.8)
        ax.set_title(f"Flip {channels[src_ch]} bit → Secret", fontsize=10)
        ax.set_xlabel("Bit position")
        ax.set_ylabel("Mean pixel change")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([str(b) for b in bit_positions])
        if col == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Bit Sensitivity: Effect of Flipping a Single Bit in Container Image\n"
                 "(Top: effect on container itself — control. Bottom: effect on recovered secret)",
                 fontsize=12)
    plt.tight_layout()
    bit_path = os.path.join(args.output_dir, "bit_sensitivity.png")
    plt.savefig(bit_path, dpi=150, bbox_inches="tight")
    print(f"Bit sensitivity analysis saved to {bit_path}")

    # --- LSB steganalysis (Paper: StegExpose / detection resistance) ---
    # Chi-square test: LSB stego equalizes pairs (2k,2k+1) -> low chi2.
    # Deep stego does NOT use LSB -> expect similar chi2 for covers and containers.
    print("\nRunning LSB steganalysis (chi-square, StegExpose-style)...")
    cover_uint8 = (np.clip(input_C, 0, 1) * 255).astype(np.uint8)
    container_uint8 = (np.clip(container_images, 0, 1) * 255).astype(np.uint8)

    cover_scores = [chi_square_lsb_score(cover_uint8[i]) for i in range(len(cover_uint8))]
    container_scores = [chi_square_lsb_score(container_uint8[i]) for i in range(len(container_uint8))]

    mean_cover = np.mean(cover_scores)
    mean_container = np.mean(container_scores)
    std_cover = np.std(cover_scores)
    std_container = np.std(container_scores)

    print(f"  Covers (clean):     χ² = {mean_cover:.1f} ± {std_cover:.1f}")
    print(f"  Containers (stego): χ² = {mean_container:.1f} ± {std_container:.1f}")
    print("  → Similar χ² means LSB detector cannot distinguish (Deep stego ≠ LSB)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(cover_scores, bins=30, alpha=0.7, color="#3498db", edgecolor="white", label="Covers")
    axes[0].hist(container_scores, bins=30, alpha=0.7, color="#e74c3c", edgecolor="white", label="Containers")
    axes[0].set_xlabel("Chi-square score (higher = less LSB-like)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("LSB Steganalysis: χ² Distribution")
    axes[0].legend()

    axes[1].bar([0], [mean_cover], yerr=[std_cover], color="#3498db", capsize=5, label="Covers")
    axes[1].bar([1], [mean_container], yerr=[std_container], color="#e74c3c", capsize=5, label="Containers")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Covers\n(clean)", "Containers\n(deep stego)"])
    axes[1].set_ylabel("Mean χ² score")
    axes[1].set_title("LSB Detector Fails: Deep Stego ≠ LSB")
    axes[1].legend()

    plt.suptitle("StegExpose-Style LSB Detection: Deep Steganography Evades LSB Detectors",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    steg_path = os.path.join(args.output_dir, "lsb_steganalysis.png")
    plt.savefig(steg_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"LSB steganalysis saved to {steg_path}")


if __name__ == "__main__":
    main()
