import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.preprocessing import normalize, scale


class CustomDataset(Dataset):

    '''
    Parameters:
        * feature1kb_file_path: path of a file including feature values of bins(1kb)
        * clusters_file_path: path of a file including clusters labels of bins(100kb)
        * hic_embedding_file_path: path of a file including hic_embedding values of bins(100kb)
        * scaler: whether to scale feature values, i.e., make each column to have mean 0 and variance 1
        * signal_resolution: signal resolution to use. one of 100kb or 1kb
    '''

    def __init__(self, feature1kb_file_path, clusters_file_path, hic_embedding_file_path, scaler=True, signal_resolution='1kb'):
        self.signal_resolution = signal_resolution
        self.np_feature1kb = np.load(feature1kb_file_path)
        self.df_clusters = pd.read_csv(clusters_file_path)
        self.df_hic_embedding = pd.read_csv(hic_embedding_file_path, header=None)

        self.X = self.np_feature1kb
    
        if scaler:
            self.X = scale(self.X.reshape(-1, self.X.shape[2]), axis=0).reshape(self.X.shape[0], self.X.shape[1], self.X.shape[2])

        self.y = self.df_hic_embedding.values
        self.chr = self.df_clusters['chr'].values
        self.pos = self.df_clusters['pos'].values
        self.idx = self.df_clusters.index.values

    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        chr, pos, idx = self.chr[index], self.pos[index], self.idx[index]

        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        return (X, y, chr, pos, idx)
    
    

def get_dataloader(feature1kb_file_path, clusters_file_path, hic_embedding_file_path, mode, shuffle=True, scaler=True, batch_size=128, random_seed=0, signal_resolution='1kb'):
    dataset = CustomDataset(feature1kb_file_path, clusters_file_path, hic_embedding_file_path, scaler, signal_resolution)
    df = dataset.df_clusters
    val_indices = df[df['chr'] % 2 == 0].sample(frac=0.4, random_state=random_seed).index # 40% of even chromosomes
    train_indices = df.index.difference(val_indices)

    if mode == 'train':
        train_subset = Subset(dataset, train_indices)
        dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
    elif mode == 'val':
        val_subset = Subset(dataset, val_indices)
        dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    elif mode == 'test':
        test_subset = Subset(dataset, val_indices) # same as val
        dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    elif mode == 'all':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        raise ValueError(f'mode must be one of train, val, test, or all, but got {mode}')
    
    return dataloader







def main():
    pass

if __name__ == '__main__':
    main()
