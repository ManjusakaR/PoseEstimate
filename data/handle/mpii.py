import torch.utils.data as data
import numpy as np
from h5py import File
import cv2
from utils.utils import Rnd, Flip, ShuffleLR
import ref as ref
from data.handle.img import Crop, DrawGaussian, Transform
import os

image_path='/home/wrruir/Downloads/images'
h5_path=os.getcwd()+"/data/mpii/annot/train.h5"

class MPII(data.Dataset):
    def __init__(self, split):
        print('==> initializing 2D {} data.'.format(split))
        annot = {}
        tags = ['imgname','part','center','scale']
        f = File(h5_path, 'r')
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        print('Loaded 2D {} {} samples'.format(split, len(annot['scale'])))

        self.split = split
        self.annot = annot

    def LoadImage(self, index):
        #print(str(self.annot['imgname'][index])[2:-1])
        path = '{}/{}'.format(image_path, self.annot['imgname'][index].decode())
        #print(path)
        img = cv2.imread(path)
        return img

    def GetPartInfo(self, index):
        pts = self.annot['part'][index].copy()
        c = self.annot['center'][index].copy()
        s = self.annot['scale'][index]
        s = s * 200
        return pts, c, s

    def __getitem__(self, index):
        img = self.LoadImage(index)
        pts, c, s = self.GetPartInfo(index)
        r = 0

        if self.split == 'train':
            s = s * (2 ** Rnd(ref.scale))
            r = 0 if np.random.random() < 0.6 else Rnd(ref.rotate)
        inp = Crop(img, c, s, r, ref.inputRes) / 256.
        out = np.zeros((ref.nJoints, ref.outputRes, ref.outputRes))
        for i in range(ref.nJoints):
            if pts[i][0] > 1:
                pt = Transform(pts[i], c, s, r, ref.outputRes)
                out[i] = DrawGaussian(out[i], pt, ref.hmGauss)
        if self.split == 'train':
            if np.random.random() < 0.5:
                inp = Flip(inp)
                out = ShuffleLR(Flip(out))
            inp[0] = np.clip(inp[0] * (np.random.random() * (0.4) + 0.6), 0, 1)
            inp[1] = np.clip(inp[1] * (np.random.random() * (0.4) + 0.6), 0, 1)
            inp[2] = np.clip(inp[2] * (np.random.random() * (0.4) + 0.6), 0, 1)
            meta = np.zeros(1)
        else:
            meta = {'index' : index, 'center' : c, 'scale' : s, 'rotate': r}

        return inp, out, meta

    def __len__(self):
        return len(self.annot['scale'])

