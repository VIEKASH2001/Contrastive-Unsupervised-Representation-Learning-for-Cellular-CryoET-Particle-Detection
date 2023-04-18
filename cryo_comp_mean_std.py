import mrcfile as mrc
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.morphology as morph

mrc_path = '/home/aldb/dataset/viekash/data1/'
save_dir = './output/'


data = []

for mrc_i in range(4):

    imgpath = '%sreconstruction_image_%d.mrc' % (mrc_path, mrc_i)

    with mrc.open(imgpath, permissive=True) as f:
        tomo_data = f.data

    # for i in range(0, 512, 20):
    #     plt.imshow(tomo_data[i])
    #     plt.show()
    n_channels = 80
    tomo_data = tomo_data[256-n_channels: 256+n_channels+1]
    print(mrc_i, tomo_data.mean(), tomo_data.std())


    data.append(tomo_data)

data = np.concatenate(data, -1)
print(data.shape)

data = data.reshape(-1)

print(data.mean(), data.std())
