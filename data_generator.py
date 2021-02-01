import scipy.io as sio
import copy
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import torch

class DataGenerator():

    def __init__(self, DataFileName = None, GTFileName = None, hwz = 9, CL_num = 20, NL_num = 12, dim = 3):
        matfn1 = sio.loadmat(DataFileName)
        matfn2 = sio.loadmat(GTFileName)
        self.data = matfn1[list(matfn1.keys())[-1]]
        self.groundtruth = matfn2[list(matfn2.keys())[-1]]
        self.class_num = np.max(self.groundtruth)
        self.in_channel = self.data.shape[-1]
        self.training = []
        self.testing = []
        self.ClassDict = {}
        self.usage = []
        self.hwz = hwz
        self.CL_num = CL_num
        self.NL_num = NL_num
        self.dim = dim
    
    def pca_transform(self, data, nc = 3, verbosity = False):
        pca = PCA(n_components= nc)
        data = pca.fit_transform(data)
        if verbosity:
            plt.figure()
            plt.imshow(data)
            plt.show()
        return data

    def RemoveFromList(self, src, target):
        for i in range(len(src)):
            if np.array_equal(src[i], target):
                src.pop(i)
                break
        return src

    def start(self):
        self.FormClassDict()
        self.GenerateTrainingSet(self.CL_num, self.NL_num)
        self.GenerateTestingSet()

        return

    def FormClassDict(self):

        # ClassDict is a dictionary with key = class, value = [spatial info, spectrum info]
        # Step 1. Copy Data as spatial_info to get spatial feature
        temp = copy.deepcopy(self.data)
        temp = temp.reshape((-1, temp.shape[-1]))
        temp = scale(temp)
        # temp = self.pca_transform(temp, nc = self.dim)
        self.in_channel = temp.shape[-1]
        info = temp.reshape((self.data.shape[0], self.data.shape[1], -1))
        # Data padding
        paddingData = np.zeros((info.shape[0] + 2 * self.hwz + 1, info.shape[1] + 2 * self.hwz + 1, info.shape[2]))
        paddingData[self.hwz: info.shape[0] + self.hwz, self.hwz: info.shape[1] + self.hwz] = info[:, :]
        # Spatial & Spectrum information extraction
        for x in range(self.groundtruth.shape[0]):
            for y in range(self.groundtruth.shape[1]):
                if self.groundtruth[x, y] == 0:
                    continue
                if self.groundtruth[x, y] not in self.ClassDict:
                    self.ClassDict[self.groundtruth[x, y]] = []
                self.ClassDict[self.groundtruth[x, y]].append(paddingData[x: x + 2 * self.hwz + 1, y : y + 2 * self.hwz + 1, :].transpose(2,0,1).astype(np.float32))
        return

    def GenerateTrainingSet(self, CL_NUM, NL_NUM):
        # In each tuple (Spatial info, Spectrum info, fake label, true_label)
        self.training = []
        for key in self.ClassDict.keys():
            # Pick NL_NUM()
            samples = random.sample(self.ClassDict[key], CL_NUM)
            for i in range(CL_NUM):
                self.training.append((samples[i], key - 1, key - 1))
                self.ClassDict[key] = self.RemoveFromList(self.ClassDict[key], samples[i])
            for j in range(NL_NUM):
                EL = list(range(1, self.class_num + 1))
                EL.remove(key)
                prob = 1 / len(EL) * np.ones((len(EL)))
                truth = np.random.choice(EL, size = 1, p = prob)[0]
                true_sample = random.sample(self.ClassDict[truth], 1)[0]
                self.training.append((true_sample, key - 1, truth - 1))
                self.ClassDict[truth] = self.RemoveFromList(self.ClassDict[truth], true_sample)
        return
    
    def GenerateTestingSet(self):
        # Testing set is a set of tuples (Spatial info, Spectrum info, True Label)
        self.testing = []
        for key in self.ClassDict.keys():
            for i in range(len(self.ClassDict[key])):
                self.testing.append((self.ClassDict[key][i], key - 1, key - 1))
        return

    def getshape(self):
        print('Training set has {} samples, there are {} elements in each tuple. \n\
                spatial information is {}'.format(len(self.training), 
                                                  len(self.training[0]),
                                                  self.training[0][0].shape))
        print('Testing set has {} samples, there are {} elements in each tuple. \n\
                spatial information is {}'.format(len(self.testing), 
                                                  len(self.testing[0]),
                                                  self.testing[0][0].shape))        
        return

    def add_noise(self, number):

        for key in self.ClassDict.keys():
            for i in range(number):
                EL = list(range(1, self.class_num + 1))
                EL.remove(key)
                prob = 1 / len(EL) * np.ones((len(EL)))
                truth = np.random.choice(EL, size = 1, p = prob)[0]
                true_sample = random.sample(self.ClassDict[truth], 1)[0]
                self.training.append((true_sample, key - 1, truth - 1))
                self.ClassDict[truth] = self.RemoveFromList(self.ClassDict[truth], true_sample)
                self.GenerateTestingSet()
        self.NL_num += number
        return

    def to_tensor(self):

        training_tensor = []
        testing_tensor  = []
        for i in range(len(self.training)):
            training_tensor.append((torch.from_numpy(self.training[i][0]), 
                                    torch.tensor(self.training[i][1]),
                                    torch.tensor(self.training[i][2])))
        for i in range(len(self.testing)):
            testing_tensor.append((torch.from_numpy(self.testing[i][0]), 
                                   torch.tensor(self.testing[i][1]),
                                   torch.tensor(self.testing[i][2])))

        return training_tensor, testing_tensor