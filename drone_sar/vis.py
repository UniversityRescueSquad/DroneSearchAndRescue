import cv2
from PIL import Image
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


def draw_prediction(pil, pred):
    im = np.array(pil)

    for x0, y0, w, h in pred:
        x1, y1 = x0 + w, y0 + h
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        im = cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 0), 3)
    im = Image.fromarray(im)
    # im.thumbnail((2000, 2000))

    return im
