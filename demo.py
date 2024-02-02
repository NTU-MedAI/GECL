from data_loader.data_loaders import QM93DDataLoader
from torch.utils.data import DataLoader
from module.molgraph_data import MolGraphDataset, molgraph_collate_fn
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from module.spherenet import SphereNet
from module.AMDE_implementations import Graph_encoder
from model.model import CombineModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(batch_size=10):

    ds = MolGraphDataset(path='/Data/Original/Important/qm9.csv')
    l = DataLoader(ds,shuffle=False,
                   batch_size=batch_size,drop_last=True, collate_fn=molgraph_collate_fn)

    for batch in l:
        adj_1, nd_1, ed_1, targets, d1, mask_1 = batch
        print("ADJ_AMDE:",adj_1.shape)
        print("ND_AMDE:",nd_1.shape)
        print("ED_AMDE:",ed_1.shape)
        print("D_AMDE:",d1.shape)
        print("MASK_AMDE:", mask_1.shape)
    #
    #     # with torch.no_grad():
    #     #     out = net(adj_1, nd_1, ed_1, d1, mask_1)
    #     #     print(out.shape)
    #
        break
    # q = QM93DDataLoader(train_batch_size=batch_size,valtest_batch_size=1)
    # spNet = SphereNet(out_channels=20).to(device)
    # amdeNet = Graph_encoder(node_features_1=75,
    #                     edge_features_1=4,
    #                     message_size=25,
    #                     message_passes=2,
    #                     out_features=10).to(device)
    # combNet = CombineModel(batch_size = batch_size).to(device)
    # for idx,batch_data in enumerate(q.train_loader):
    #     batch_data = batch_data.to(device)
    #     out = combNet(batch_data)
    #     print(out.shape)
        # adj,nd,ed,d, mask = batch_data.adj,batch_data.nd,batch_data.ed,batch_data.d, batch_data.mask
        # adj = adj.reshape(batch_size,-1,adj.shape[1]).to(device)
        # nd = nd.reshape(batch_size,-1,nd.shape[1]).to(device)
        # ed = ed.reshape(batch_size,-1,ed.shape[1],ed.shape[2]).to(device)
        # d = d.reshape(batch_size,-1).to(device)
        # mask = mask.reshape(batch_size,-1).to(device)
        #
        # (e, v, u) = spNet(batch_data)
        # out = amdeNet(adj,nd,ed,d, mask)
        # result = torch.concat([out,u],dim=1)
        # print(result.shape)
        # print("amdeOut:",out.shape)
        # print("e.shape:",e[0].shape,e[1].shape)
        # print("v.shape:", v.shape)
        # print("u.shape:", u.shape)
        break


def equal(a,b):
    return a-b < 0.00001
def test():
    qm = np.load('/home/ntu/PycharmProjects/CY/Data/Original/qm9/raw/qm9_eV.npz')
    data = pd.read_csv('/Data/Original/Important/qm9.csv')
    # keeprows = []
    # i=0
    # for _,row in tqdm(data.iterrows()):
    #     if equal(qm['A'][i],row['A']) and equal(qm['B'][i],row['B']) and equal(qm['C'][i],row['C']):
    #         keeprows.append(i)
    #         i+=1
    # print(len(keeprows))
    keeprows = list(np.load('/Data/Original/Important/keeprows.npy'))
    print(list(data.iloc[keeprows]['smiles']))

def tensorFill():
    a = torch.tensor([
        [1,0,1],
        [0,1,0],
        [0,0,1]
    ])
    b = torch.zeros((9,9))
    b[:a.shape[0],:a.shape[1]] = a
    a = torch.tensor([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])
    print(b)


if __name__ == '__main__':
    for batch_size in[23,129]:
        print("BATCH_SIZE:",batch_size)
        main(batch_size)