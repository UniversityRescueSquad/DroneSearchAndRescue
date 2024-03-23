import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def show_detections(pil, colored_boxes_dict, size=10, im_opacity=0.5):
    np_pil = pil

    fig, ax = plt.subplots(dpi=100, figsize=(size, size))
    plt.tight_layout()
    ax.imshow(np_pil, alpha=im_opacity)

    for color, boxes in colored_boxes_dict.items():
        for x, y, w, h in boxes:
            ax.add_patch(
                Rectangle((x, y), w, h, fill=None, edgecolor=color, linewidth=1)
            )

    plt.close()

    return fig
