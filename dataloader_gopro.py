from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler
import os
from PIL import Image
from random import sample 
class GOPRO(Dataset):
    def __init__(self,data_path,is_guide = False,guide_path = 'HIDE_dataset'):
        scenes = os.listdir(data_path)
        self.path_images = []
        self.path_gts = []
        for scence in scenes:
            path = os.path.join(data_path,scence)
            blur_path = os.path.join(path,'blur')
            gt_path = os.path.join(path,'sharp')
            images = os.listdir(blur_path)
            gt = os.listdir(gt_path)
            for idx,_ in enumerate(images):
                self.path_images.append(os.path.join(blur_path,images[idx]))
                self.path_gts.append(os.path.join(gt_path,gt[idx]))
        self.trans = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.is_guide = is_guide
        self.path_guides = []

        if self.is_guide:
            guide_paths = os.path.join(guide_path,'GT')
            scences = os.listdir(guide_path)
            for scence in scences:
                # gt_scene = str.split(scence,'.')[0] + '.png'
                guide_path = os.path.join(gt_paths,scence)
                # for idx,_ in enumerate(images):
                self.path_guides.append(guide_path)

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self,idx):
        image = Image.open(self.path_images[idx])
        gt = Image.open(self.path_gts[idx])
        image = self.trans(image)
        gt = self.trans(gt)

        ### TESTING NEW GUIDANCE ###
        if self.is_guide:
            guide = Image.open(sample(self.path_guides,1)[0])
            guide = self.trans(guide)

        else:
            guide = Image.open(sample(self.path_gts,1)[0])
            guide = self.trans(guide)

        return image,gt,guide


def load_data(batchsize:int, numworkers:int, data_path:str = None,shuffle:bool = True,is_guide:bool = False) -> tuple[DataLoader, DistributedSampler]:
    data_train = GOPRO(data_path,is_guide=is_guide)
    # sampler = DistributedSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        shuffle=shuffle
                        # sampler = sampler,
                        # drop_last = True
                    )
    return trainloader

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5
