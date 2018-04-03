from h5py import File
import numpy as np
import scipy.io as sio
import cv2


'''
annot={}
tags=['imgname','part','center','scale']
f=File('val.h5')
for tag in tags:
    annot[tag]=np.asarray(f[tag]).copy()
f.close()
print(annot['imgname'])


data=sio.loadmat('mpii_human_pose_v1_u12_1.mat')
print(type(data["RELEASE"]))
print(data['RELEASE'].dtype)
print(data['RELEASE']['annolist'][0][0])
'''


