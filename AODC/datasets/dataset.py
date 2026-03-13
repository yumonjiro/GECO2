# -*- coding: utf-8 -*-

import torch.utils.data as data
import os
from PIL import Image
import json
from .utils import NormalSample, jpg2id
import torch
import numpy as np

class FSC147(data.Dataset):
    def __init__(self, root_path, mode, class_remove=False):
        super().__init__()
        with open(os.path.join(root_path, 'Train_Test_Val_FSC_147.json')) as f:
            imglist = json.load(f)[mode]
        with open(os.path.join(root_path, 'fsc147_384x576.json')) as f:
            samples = json.load(f)
        if class_remove:
            self.imgids = []
            for imgf in imglist:
                imgid = jpg2id(imgf)
                if samples[imgid]["category"] != "cars":
                    self.imgids.append(imgid)
        else:
            self.imgids = [jpg2id(imgf) for imgf in imglist]
        self.samples = {idx: samples[idx] for idx in self.imgids}

        self.it2cat = dict()
        with open(os.path.join(root_path, 'ImageClasses_FSC147.txt')) as f:
            catdict = dict()
            for line in f.read().strip().split('\n'):
                a, b = line.split('.jpg')
                a, b = a.strip(), b.strip()
                if b not in catdict:
                    catdict[b] = len(catdict) + 1
                self.it2cat[a] = b

        self.root_path = root_path

        self.normalfunc = NormalSample()

        self.can_h = 384
        self.can_w = 576

    def __getitem__(self, index):
        imgid = self.imgids[index]

        sample = self.getSample(imgid)
        category = 'a photo of ' + self.it2cat[imgid]

        return (*sample, imgid, category)

    def __len__(self):
        return len(self.imgids)

    def getSample(self, imgid):
        sample = self.samples[imgid]
        image = Image.open(os.path.join(self.root_path, sample['imagepath']))
        w, h = image.size

        points = torch.tensor(sample['points']).round().long()  # N x (w, h)
        boxes = torch.clip(torch.tensor(sample['boxes'][:3]).view(3, 4).round().long(),
                           min=0)  # 3 x ((left, top), (right, bottom))
        dotmap = np.zeros((1, h, w), dtype=np.float32)
        points[:, 1] = torch.clip(points[:, 1], min=0, max=h - 1)
        points[:, 0] = torch.clip(points[:, 0], min=0, max=w - 1)
        dotmap[0, points[:, 1], points[:, 0]] = 1

        image, dotmap = self.normalfunc(image, dotmap)
        for i, box in enumerate(boxes):
            l, t, r, b = box
            b, r = max(t + 1, b), max(l + 1, r)
            boxes[i] = torch.tensor([l, t, r, b])

        return image, boxes, dotmap

    @staticmethod
    def collate_fn(samples):
        images, boxes, dotmaps, imgids, categories = zip(*samples)
        images = torch.stack(images, dim=0)
        shot = boxes[0].shape[0]
        index = torch.arange(images.size(0)).view(-1, 1).repeat(1, shot).view(-1, 1)
        boxes = torch.cat([index, torch.cat(boxes, dim=0)], dim=1)
        dotmaps = torch.stack(dotmaps, dim=0)
        return images, boxes, dotmaps, imgids, list(categories)


class CARPK(data.Dataset):
    def __init__(self, root_path):
        super().__init__()
        file = open('datasets/CARPK_devkit/' + 'test' + '.json', 'r', encoding='utf-8')
        self.imglist = []
        for line in file.readlines():
            dic = json.loads(line)
            self.imglist.append(dic)

        self.root_path = root_path

        self.normalfunc = NormalSample()


    def __getitem__(self, index):
        imginfo = self.imglist[index]

        sample = self.getSample(imginfo)
        return (*sample, imginfo["filename"], 'a photo of cars')

    def __len__(self):
        return len(self.imglist)

    def getSample(self, imginfo):
        img = Image.open(os.path.join(self.root_path, 'CARPK_devkit/data/Images/' + imginfo["filename"]))
        w, h = img.size

        all_boxes = imginfo["boxes"]
        boxes = torch.tensor(all_boxes[0: 2])
        dotmap = np.zeros((1, h, w), dtype=np.float32)
        for box in all_boxes:
            dotmap[0, int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)] = 1

        img, density_map = self.normalfunc(img, dotmap)
        for i, box in enumerate(boxes):
            t, l, b, r = box
            b, r = max(t + 1, b), max(l + 1, r)
            boxes[i] = torch.tensor([l, t, r, b])

        return img, boxes, density_map
