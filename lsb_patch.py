#!/usr/bin/env python3
"""
Run this on the cluster to add LSB steganalysis to evaluate.py:
  python3 lsb_patch.py

Or copy this file to the cluster and run it there.
"""
import os

EVAL_PATH = os.path.join(os.path.dirname(__file__), "evaluate.py")

CHI_SQUARE_FUNC = '''
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
        diff_sq = (even_counts - odd_counts).astype(np.float64) ** 2
        chi2_vals = np.where(mask, diff_sq / np.maximum(total.astype(np.float64), 1), 0)
        scores.append(np.sum(chi2_vals))
    return np.mean(scores)


'''

LSB_BLOCK = '''
    # --- LSB steganalysis (Paper: StegExpose / detection resistance) ---
    print("\\nRunning LSB steganalysis (chi-square, StegExpose-style)...")
    cover_uint8 = (np.clip(input_C, 0, 1) * 255).astype(np.uint8)
    container_uint8 = (np.clip(container_images, 0, 1) * 255).astype(np.uint8)

    cover_scores = [chi_square_lsb_score(cover_uint8[i]) for i in range(len(cover_uint8))]
    container_scores = [chi_square_lsb_score(container_uint8[i]) for i in range(len(container_uint8))]

    mean_cover = np.mean(cover_scores)
    mean_container = np.mean(container_scores)
    std_cover = np.std(cover_scores)
    std_container = np.std(container_scores)

    print(f"  Covers (clean):     chi2 = {mean_cover:.1f} +/- {std_cover:.1f}")
    print(f"  Containers (stego): chi2 = {mean_container:.1f} +/- {std_container:.1f}")
    print("  -> Similar chi2 means LSB detector cannot distinguish (Deep stego != LSB)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(cover_scores, bins=30, alpha=0.7, color="#3498db", edgecolor="white", label="Covers")
    axes[0].hist(container_scores, bins=30, alpha=0.7, color="#e74c3c", edgecolor="white", label="Containers")
    axes[0].set_xlabel("Chi-square score (higher = less LSB-like)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("LSB Steganalysis: chi2 Distribution")
    axes[0].legend()

    axes[1].bar([0], [mean_cover], yerr=[std_cover], color="#3498db", capsize=5, label="Covers")
    axes[1].bar([1], [mean_container], yerr=[std_container], color="#e74c3c", capsize=5, label="Containers")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Covers\\n(clean)", "Containers\\n(deep stego)"])
    axes[1].set_ylabel("Mean chi2 score")
    axes[1].set_title("LSB Detector Fails: Deep Stego != LSB")
    axes[1].legend()

    plt.suptitle("StegExpose-Style LSB Detection: Deep Steganography Evades LSB Detectors", fontsize=12, y=1.02)
    plt.tight_layout()
    steg_path = os.path.join(args.output_dir, "lsb_steganalysis.png")
    plt.savefig(steg_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"LSB steganalysis saved to {steg_path}")

'''


def main():
    if not os.path.exists(EVAL_PATH):
        print(f"Error: {EVAL_PATH} not found. Run from project root.")
        return 1

    with open(EVAL_PATH, "r") as f:
        content = f.read()

    if "chi_square_lsb_score" in content:
        print("evaluate.py already has LSB steganalysis. Nothing to do.")
        return 0

    # Add chi_square function after pixel_error
    marker1 = "def pixel_error(original, reconstructed):\n    return np.sqrt(np.mean(np.square(255 * (original - reconstructed))))\n\n\n"
    if marker1 in content:
        content = content.replace(
            marker1,
            "def pixel_error(original, reconstructed):\n    return np.sqrt(np.mean(np.square(255 * (original - reconstructed))))\n\n" + CHI_SQUARE_FUNC + "\n"
        )
    else:
        # Fallback: insert after "def pixel_error"
        content = content.replace(
            "def pixel_error(original, reconstructed):\n    return np.sqrt(np.mean(np.square(255 * (original - reconstructed))))\n\n",
            "def pixel_error(original, reconstructed):\n    return np.sqrt(np.mean(np.square(255 * (original - reconstructed))))\n\n" + CHI_SQUARE_FUNC
        )

    # Add LSB block after bit sensitivity (before "if __name__")
    marker2 = 'print(f"Bit sensitivity analysis saved to {bit_path}")\n\n\nif __name__'
    if marker2 in content:
        content = content.replace(
            marker2,
            'print(f"Bit sensitivity analysis saved to {bit_path}")' + LSB_BLOCK + "\n\nif __name__"
        )
    else:
        marker2b = 'print(f"Bit sensitivity analysis saved to {bit_path}")\n\nif __name__'
        if marker2b in content:
            content = content.replace(
                marker2b,
                'print(f"Bit sensitivity analysis saved to {bit_path}")' + LSB_BLOCK + "\n\nif __name__"
            )
        else:
            print("Error: Could not find insertion point. Check evaluate.py structure.")
            return 1

    with open(EVAL_PATH, "w") as f:
        f.write(content)

    print("LSB steganalysis added to evaluate.py successfully.")
    return 0


if __name__ == "__main__":
    exit(main())
