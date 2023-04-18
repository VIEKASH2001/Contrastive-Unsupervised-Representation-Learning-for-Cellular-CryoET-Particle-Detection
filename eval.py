import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
## Load data
import mrcfile as mrc
import warnings
from sklearn.cluster import KMeans
import numpy as np
from models_simclr.cifar_resnet import resnet18
import re
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
warnings.simplefilter('ignore') # to mute some warnings produced when opening the tomos

## Evaluation -
## Extract uniform patches from image of fixed size (fold/unfold pytorch)
## Representation for each patch ([128,]) : (100,128)
## 4096 patches -> 128 dimensions each (Say patch size = 64,64)
## For each patch -> A label will be there, similar patches will be clustered together

denoised_data_path = r'./data/shrec/data/reconstruction_image_0.mrc'
slc = 200
patch_size = 96 # 64 or 96
overlap_size = 92 # (0 or 60), (0 or 92)
model_path = r'./logs/shrec_aug_test/model_1200.pt'
n_clusters = 2

ol = "no_overlap"
if overlap_size > 0:
    ol = "overlap"
epoch = re.findall('\d+',model_path)[0]
img_name = denoised_data_path.split('/')[-1].split('.')[0]
exp = f'sk_large_{patch_size}_{n_clusters}_{ol}_{str(epoch)}_{img_name}'

class Patchify:
    def __init__(self, patch_size, overlap_size):
        self.patch_size = patch_size
        self.overlap_size = self.patch_size - overlap_size

    def __call__(self, x):
        x = x.unsqueeze(0)
        b, c, h, w = x.size()

        # patch up the images
        # (b, c, h, w) -> (b, c*patch_size, L)
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.overlap_size)

        # (b, c*patch_size, L) -> (b, nb_patches, width, height)
        x = x.transpose(2, 1).contiguous().view(b, -1, self.patch_size, self.patch_size)

        # reshape to have (b x patches, c, h, w)
        x = x.view(-1, c, self.patch_size, self.patch_size)

        x = x.squeeze(0)

        return x

# To pick n evenly spaced 
def GetSpacedElements(array, numElems = 4096):
    out = np.round(np.linspace(0, len(array)-1, numElems)).astype(int)
    return out

activation = {}
def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    return new_state_dict

with mrc.open(denoised_data_path,permissive=True) as f:
    data = f.data

raw_img = torch.tensor(data[slc]).unsqueeze(0)
## 96x96 -> 9216 patches
getPatches = Patchify(patch_size=patch_size,overlap_size=overlap_size)
img_patched = getPatches(raw_img)
if overlap_size != 0:
    if patch_size == 96:
        numElems = 9216
    else:
        numElems = 4096
    spaced_idx = GetSpacedElements([i for i in range(img_patched.shape[0])], numElems = numElems)
    img_patched = img_patched[spaced_idx] ## Pick overlapping 4096 patches of 64x64 each


net = resnet18(zero_init_residual=True, num_classes=1024)
net.layer4[1].conv2.register_forward_hook(get_activation('conv2'))
net = net.to('cuda:0')
net.eval()
# net = nn.DataParallel(net)
checkpoint = torch.load(model_path)
new_state = remove_data_parallel(checkpoint['net'])
net.load_state_dict(new_state)
input_data = img_patched.repeat([1,3,1,1])

eval_dataset = TensorDataset(input_data)
eval_dataloader = DataLoader(eval_dataset,batch_size=64)
for batch_idx,batched_sample in enumerate(eval_dataloader):
    net(batched_sample[0].cuda()) # Forward pass
    features = activation['conv2'] # Get activation before MLP head
    features = features.mean([2, 3]) # Output : (bs,512) 

    if batch_idx == 0:
        colated_feats = features
    else:
        colated_feats = torch.cat((colated_feats,features),0)



features = colated_feats.view(-1, 512)
# normalize the features in case the kmeans class does not normalize it!
features = F.normalize(features, 2, 1)
features = features.detach().cpu().numpy()
print(features.shape)
kmeans = KMeans(n_clusters=n_clusters,max_iter=1000,tol=1e-20).fit(features)
cluster_ids_x = kmeans.labels_
# for each patch, we will get a label in [0, n_clusters-1]
if overlap_size == 0:
    if patch_size == 64:
        view_size = 8
    if patch_size==96:
        view_size=5
else:
    view_size = patch_size

a = cluster_ids_x.reshape(view_size, view_size)
a = (a - a.min()) / (a.max() - a.min())

fig,ax = plt.subplots(1,2,figsize=(10,10))
ax[0].imshow(a)
ax[1].imshow(raw_img[0],cmap='Greys_r')
plt.savefig('./plots/{}.png'.format(exp),dpi=80)