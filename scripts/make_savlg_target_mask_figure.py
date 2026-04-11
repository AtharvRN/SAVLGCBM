from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT = Path("docs/figures/savlg_target_mask_example.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig = plt.figure(figsize=(9.2, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.18, 1.0], wspace=0.18)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_mask = fig.add_subplot(gs[0, 2])

    # Left: image-plane schematic with grid and bbox.
    ax_img.set_xlim(0, 4)
    ax_img.set_ylim(0, 4)
    ax_img.set_aspect("equal")
    ax_img.set_xticks(np.arange(0, 5, 1))
    ax_img.set_yticks(np.arange(0, 5, 1))
    ax_img.grid(color="#B8C2CC", linewidth=1.0)
    ax_img.set_xticklabels([])
    ax_img.set_yticklabels([])
    ax_img.tick_params(length=0)
    for spine in ax_img.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#444444")

    # Soft background hinting at an image.
    bg = np.zeros((240, 240, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:240, 0:240]
    bg[..., 0] = 0.96 - 0.12 * (yy / 240.0)
    bg[..., 1] = 0.97 - 0.08 * (xx / 240.0)
    bg[..., 2] = 0.99 - 0.10 * ((xx + yy) / 480.0)
    ax_img.imshow(bg, extent=(0, 4, 0, 4), origin="lower", zorder=0)

    # A simple "bird-like" blob for context.
    bird = np.array(
        [
            [0.7, 2.0],
            [1.2, 2.5],
            [1.9, 2.65],
            [2.8, 2.4],
            [3.0, 2.0],
            [2.45, 1.55],
            [1.5, 1.4],
            [0.95, 1.6],
        ]
    )
    ax_img.fill(bird[:, 0], bird[:, 1], color="#D9E7F5", alpha=0.95, zorder=1)
    ax_img.plot(bird[:, 0], bird[:, 1], color="#4A6A8A", linewidth=1.5, zorder=2)

    # Bounding box that partially overlaps several cells.
    bbox = Rectangle((1.15, 1.2), 1.65, 1.75, fill=False, ec="#C43C2F", lw=2.4, zorder=3)
    ax_img.add_patch(bbox)
    ax_img.text(
        1.2,
        3.07,
        "GT concept box",
        color="#C43C2F",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax_img.set_title("Image plane + supervision grid", fontsize=11, pad=14)

    # Middle arrow.
    ax_mid.axis("off")
    arrow = FancyArrowPatch(
        (0.08, 0.5),
        (0.92, 0.5),
        transform=ax_mid.transAxes,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.0,
        color="#666666",
    )
    ax_mid.add_patch(arrow)
    ax_mid.text(
        0.5,
        0.62,
        "rasterize",
        transform=ax_mid.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

    # Right: target mask values.
    t_k = np.array(
        [
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.25, 0.50, 0.00],
            [0.00, 0.50, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    ax_mask.imshow(t_k, cmap="Blues", vmin=0.0, vmax=1.0, origin="upper")
    ax_mask.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax_mask.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax_mask.grid(which="minor", color="white", linewidth=1.8)
    ax_mask.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax_mask.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#444444")
    for i in range(4):
        for j in range(4):
            val = t_k[i, j]
            ax_mask.text(
                j,
                i,
                f"{val:.2f}" if val > 0 else "0",
                ha="center",
                va="center",
                fontsize=10,
                color="#0E2A47" if val < 0.7 else "white",
                fontweight="bold" if val > 0 else None,
            )
    ax_mask.set_title(r"Soft target mask $t_k$", fontsize=11, pad=14)
    ax_mask.text(
        0.5,
        -0.16,
        r"Each cell value = $\frac{\mathrm{box\ area\ inside\ patch}}{\mathrm{patch\ area}}$",
        transform=ax_mask.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#444444",
    )

    fig.suptitle(
        "Bounding-box supervision is rasterized onto the spatial concept-map grid",
        fontsize=12,
        y=0.97,
    )
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.06, right=0.98)
    fig.savefig(OUT, bbox_inches="tight")


if __name__ == "__main__":
    main()
