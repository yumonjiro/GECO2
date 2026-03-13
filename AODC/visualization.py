import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image


def visualize_output_and_save(imgid, output,config,gt, save_path, offset, target, size):
    """
        dots: Nx2 numpy array for the ground truth locations of the dot annotation
            if dots is None, this information is not available
    """
    file_path = config.DATA.DATA_PATH + '/images_384x576/' + imgid + '.jpg'
    origin_img = Image.open(file_path).convert("RGB")
    origin_img = np.array(origin_img)
    h, w, _ = origin_img.shape

    cmap = plt.cm.get_cmap('jet')
    density_map = output
    pred = output.sum().item()
    density_map = torch.nn.functional.interpolate(density_map, (h, w), mode='bilinear').squeeze(0).squeeze(
        0).numpy()

    origin_img = origin_img / origin_img.max() * 255.0
    density_map = cmap(density_map / (density_map.max()) + 1e-14) * 255.0
    density_map = density_map[:, :, 0:3] * 0.5 + origin_img * 0.5

    fig = plt.figure(dpi=800)
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax2.set_title(str(pred) + "  " + str(gt))
    ax2.imshow(density_map.astype(np.uint8))

    ax2.imshow(origin_img.astype(np.uint8))
    '''for hh in range(h):
        for ww in range(w):
            if target[0, 0, hh, ww] > 0.1:
                t_offset = offset[0, :, :, int(hh / 8), int(ww / 8)].numpy()
                x = t_offset[:, 1]
                y = t_offset[:, 0]
                ax2.scatter(x * 8 + ww, y * 8 + hh, c='blue', s=size, alpha=0.6, linewidth=0, marker='s')'''

    plt.savefig(save_path)
    plt.close()



