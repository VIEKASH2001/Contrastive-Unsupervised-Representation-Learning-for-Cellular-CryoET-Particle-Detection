import mrcfile as mrc
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.morphology as morph

mrc_path = '/home/aldb/dataset/viekash/data1/'
save_dir = './output/'

for mrc_i in range(10):

    imgpath = '%sreconstruction_image_%d.mrc' % (mrc_path, mrc_i)

    with mrc.open(imgpath, permissive=True) as f:
        tomo_data = f.data

    # for i in range(0, 512, 20):
    #     plt.imshow(tomo_data[i])
    #     plt.show()
    n_channels = 80
    ch = [256 - n_channels, 256 + n_channels]
    for idx in range(256-n_channels, 256+n_channels+1):
    # for idx in ch:

        print(mrc_i, idx)
        # idx = 191
        img = tomo_data[idx]

        # img = np.abs(img)

        # print(img.min(), img.max())

        img = (img - img.min()) / (img.max() - img.min())

        img = (img * 255).astype(np.uint8)
        imgb = cv2.GaussianBlur(img, (25, 25), cv2.BORDER_DEFAULT)

        std = imgb.reshape(-1).std()
        mean = imgb.mean()

        alpha = 1.5

        imgb[imgb < mean - std*alpha] = 0
        imgb[imgb > 0] = 1

        imgb = 1 - imgb

        # label connected components in the image
        y = morph.label(imgb, background=0)

        new_mask = np.zeros_like(imgb)

        for i in range(y.max()):
            area = (y == i).sum()
            relative_area = area * 100 / (512 * 512)

            # print(i, area, relative_area)

            if relative_area < 0.2 and relative_area > 0.01:
                new_mask[y == i] = 1

        plot = False
        if plot:
            plt.subplot(1, 4, 1)
            plt.imshow(img, cmap='gray')
            plt.subplot(1, 4, 2)
            plt.imshow(imgb, cmap='gray')
            plt.subplot(1, 4, 3)
            plt.imshow(new_mask, cmap='gray')
            plt.subplot(1, 4, 4)
            plt.imshow(img)
            plt.imshow(new_mask, cmap='jet', alpha=0.3)
            plt.show()

        # save particle mask

        file_name = save_dir + '%d_%d.jpg' % (mrc_i, idx)

        new_mask = new_mask * 255
        new_mask = new_mask.astype(np.uint8)

        cv2.imwrite(file_name, new_mask)