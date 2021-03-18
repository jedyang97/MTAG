import os
import pickle
import numpy as np
import torch
import torch.utils.data as Data
from consts import GlobalConsts as gc

class MultimodalSubdata():
    def __init__(self, name="train"):
        self.name = name
        self.text = np.empty(0)
        self.audio = np.empty(0)
        self.vision = np.empty(0)
        self.y = np.empty(0)

class MoseiDataset(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, cls="train"):
        self.root = root
        self.cls = cls
        if len(MoseiDataset.trainset.y) != 0 and cls != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data()

        if self.cls == "train":
            self.dataset = MoseiDataset.trainset
        elif self.cls == "test":
            self.dataset = MoseiDataset.testset
        elif self.cls == "valid":
            self.dataset = MoseiDataset.validset

        self.language = self.dataset.language
        self.acoustic = self.dataset.acoustic
        self.visual = self.dataset.visual
        self.y = self.dataset.y


    def load_data(self):
        if gc.data_path[-1] != '/':
            gc.data_path = gc.data_path + '/'
        dataset = pickle.load(open(gc.data_path + 'tensors.pkl', 'rb'))
        split = {'train':0, 'valid':1, 'test':2}
        gc.padding_len = dataset[0][split['test']]['glove_vectors'].shape[1]
        gc.config['text_dim'] = dataset[0][split['test']]['glove_vectors'].shape[2]
        gc.config['audio_dim'] = dataset[0][split['test']]['COAVAREP'].shape[2]
        gc.config['vision_dim'] = dataset[0][split['test']]['FACET 4.2'].shape[2]

        for ds, split_type in [(MoseiDataset.trainset, 'train'), (MoseiDataset.validset, 'valid'),
                               (MoseiDataset.testset, 'test')]:
            ds.language = torch.tensor(dataset[0][split[split_type]]['glove_vectors'].astype(np.float32)).cpu().detach()
            ds.acoustic = torch.tensor(dataset[0][split[split_type]]['COAVAREP'].astype(np.float32))
            ds.acoustic[ds.acoustic == -np.inf] = 0
            ds.acoustic = ds.acoustic.clone().cpu().detach()
            ds.visual = torch.tensor(dataset[0][split[split_type]]['FACET 4.2'].astype(np.float32)).cpu().detach()
            ds.y = torch.tensor(dataset[0][split[split_type]]['All Labels'].astype(np.float32)).cpu().detach()

    def __getitem__(self, index):
        inputLen = len(self.language[index])
        return self.language[index], self.acoustic[index], self.visual[index], \
               inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MoseiDataset(gc.data_path)
