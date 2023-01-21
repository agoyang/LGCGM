from copy import deepcopy
import os.path as osp
import torch
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import pandas as pd

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GraphConv, dense_mincut_pool, GCNConv
from torch_geometric import utils
from torch_geometric.nn import Sequential, SGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


from sklearn.metrics import normalized_mutual_info_score as NMI

from sgc import SGC
from sgc_multi import SGC as SGC1
from gcn import GCN
from deeprobust.graph.utils import normalize_adj_tensor, accuracy
from ogb.nodeproppred import PygNodePropPredDataset
from coarsening import coarsening
import argparse
import metis
import networkx as nx
import collections
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
parser.add_argument('--coarsening_ratio', type=float, default=0.1)
parser.add_argument('--out_loop', type=int, default=20)
parser.add_argument('--inner_loop', type=int, default=5)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--sgc_drop', type=float, default=0)
parser.add_argument('--num_syn', type=int, default=140)
parser.add_argument('--init_way', type=str, default='zero')
parser.add_argument('--max_out_match', type=int, default=10)
parser.add_argument('--cluster', type=int, default=10)
parser.add_argument('--hid', type=int, default=256)
args = parser.parse_args()
print(args)
torch.cuda.set_device(args.gpu_id)
# print(torch.cuda.get_device_name())
torch.manual_seed(args.seed) # for reproducibility

# Load data

dataset = args.dataset
coarsening_ratio = args.coarsening_ratio
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/new', dataset)

if dataset in ['ogbn-arxiv', 'ogbn-products']:
    dataset = PygNodePropPredDataset(dataset, path)
    data = dataset[0]
    if args.dataset == 'ogbn-arxiv':
        data.edge_index = utils.to_undirected(data.edge_index)
    elif args.dataset == 'ogbn-products':
        data.edge_index = utils.remove_self_loops(data.edge_index)[0]
    data.y = data.y.squeeze(dim=-1)
    split_idx = dataset.get_idx_split()
    data.train_mask = torch.BoolTensor([False] * data.num_nodes)
    data.train_mask[split_idx["train"]] = torch.BoolTensor([True] * len(split_idx["train"]))
    data.val_mask = torch.BoolTensor([False] * data.num_nodes)
    data.val_mask[split_idx["valid"]] = torch.BoolTensor([True] * len(split_idx["valid"]))
    data.test_mask = torch.BoolTensor([False] * data.num_nodes)
    data.test_mask[split_idx["test"]] = torch.BoolTensor([True] * len(split_idx["test"]))
elif dataset in ['Cora']:
    dataset = Planetoid(path, dataset)
    data = dataset[0]

data = data.cpu()
if args.cluster > 1:
    nx_g = utils.to_networkx(data)
    _, parts = metis.part_graph(nx_g, args.cluster)
    parts = torch.LongTensor(parts)
else:
    parts = torch.zeros(data.num_nodes)
x_syn_l = []
adj_syn_l = []
label_syn_l = []
syn_mask_l = []

def condense_cluster(cluster, gpu):
    # torch.cuda.set_device(gpu)
    nodes_in_clus = torch.arange(0, data.num_nodes)[parts == cluster]
    sub_data = data.subgraph(nodes_in_clus)

    # try:
    #     num_features = torch.load('save_coarsen/num_features_' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
    #     num_classes = torch.load('save_coarsen/num_classes_' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
    #     candidate = torch.load('save_coarsen/candidate_' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
    #     C_list = torch.load('save_coarsen/C_list_' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
    #     Gc_list = torch.load('save_coarsen/Gc_list_' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
    # except:
    num_features, num_classes, candidate, C_list, Gc_list = coarsening(sub_data, 1-coarsening_ratio, 'variation_neighborhoods')
        # torch.save(num_features, 'save_coarsen/num_features_' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
        # torch.save(num_classes, 'save_coarsen/num_classes_' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
        # torch.save(candidate, 'save_coarsen/candidate_' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
        # torch.save(C_list, 'save_coarsen/C_list_' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
        # torch.save(Gc_list, 'save_coarsen/Gc_list_' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')

    sub_data = sub_data.cuda()
    num_syn_list = [C.shape[0] for C in C_list] + [len(H.info['orig_idx']) for H in candidate[len(C_list):]]
    # print(num_syn_list)
    num_syn = [sum(num_syn_list)]
    print("synthetic nodes number:", num_syn)
    if args.init_way == 'eye':
        ini_s = torch.eye(num_syn[0], sub_data.num_nodes).T
    elif args.init_way == 'rand':
        ini_s = torch.rand((sub_data.num_nodes, num_syn[0])) / num_syn[0]
    elif args.init_way == 'zero':
        ini_s = torch.zeros((sub_data.num_nodes, num_syn[0]))
    for i in range(len(candidate)):
        H = candidate[i]
        keep = H.info['orig_idx']
        start_col = sum(num_syn_list[:i])
        if i < len(C_list):
            C = C_list[i]
            ini_s[keep, start_col : start_col + num_syn_list[i]] = torch.Tensor(C.todense()).transpose(0, 1)
        else:
            C = torch.eye(len(keep), len(keep))
            ini_s[keep, start_col : start_col + num_syn_list[i]] = C
        
    # Normalized adjacency matrix
    adj_full = torch.sparse_coo_tensor(sub_data.edge_index,
                torch.ones_like(sub_data.edge_index[0]).float(), (sub_data.num_nodes, sub_data.num_nodes))
    sub_data.edge_index, sub_data.edge_weight = gcn_norm(  
                sub_data.edge_index, sub_data.edge_weight, sub_data.num_nodes,
                add_self_loops=True, dtype=sub_data.x.dtype)
    adj_full_norm = torch.sparse_coo_tensor(sub_data.edge_index,
                sub_data.edge_weight, (sub_data.num_nodes, sub_data.num_nodes))

    #one hot label
    labels = F.one_hot(sub_data.y, dataset.num_classes).float().cuda()
    labels[~sub_data.train_mask] = torch.Tensor([0 for _ in range(dataset.num_classes)]).cuda()
    #find train nodes for each class
    class_to_idx = {}
    for c in range(dataset.num_classes):
        class_to_idx[c] = ((sub_data.y == c) & sub_data.train_mask).nonzero().squeeze()
    class Net(torch.nn.Module):
        def __init__(self,
                    n_clusters, labels):
            super().__init__()
            self.n_clusters = n_clusters
            self.labels = labels
            if isinstance(n_clusters, int):
                n_clusters = [n_clusters]
            self.s = torch.nn.ParameterList([torch.nn.Parameter(ini_s)])
            for i in range(1, len(n_clusters)):
                # self.s = torch.nn.Parameter(torch.randn(data.num_nodes, num_syn))
                self.s.append(torch.nn.Parameter(torch.randn(n_clusters[i-1], n_clusters[i])))
            # self.s = torch.nn.Parameter(torch.eye(data.num_nodes))
            # self.x = torch.nn.Parameter(torch.rand((n_clusters[-1], data.num_features)))

        def forward(self, x, adj):
            D = utils.degree(adj._indices()[0], num_nodes=sub_data.num_nodes)
            D = torch.sparse_coo_tensor(torch.arange(0, sub_data.num_nodes).repeat(2, 1).cuda(),
            D)
            # adj = adj.to_dense()
            A = deepcopy(adj)
            labels = self.labels
            syn_mask = None
            for i, s in enumerate(self.s):
                # s_norm = F.softmax(s, dim=1)
                s_norm = F.relu(s)
                # s_norm = s_norm / (s_norm.sum(dim=1).reshape(-1, 1) + 1e-8)
                # s_norm = s
                x = s_norm.transpose(0, 1) @ x
                adj = torch.spmm(adj, s_norm)
                adj = torch.spmm(adj.transpose(0, 1), s_norm).transpose(0, 1)
                # adj = s_norm.transpose(0, 1) @ adj @ s_norm
                temp = s_norm.argmax(1)
                P = F.one_hot(temp, self.n_clusters[i]).float()
                if syn_mask is not None:
                    labels = F.one_hot(labels, dataset.num_classes).float().cuda()
                    labels[~syn_mask] = torch.Tensor([0 for _ in range(dataset.num_classes)]).cuda()
                SY = P.transpose(0, 1) @ labels
                syn_mask = torch.sum(SY, dim=1).bool()
                labels = torch.argmax(SY, dim=1)
                if not i:
                    s_all = s_norm
                else:
                    s_all = s_all @ s_norm
            D = torch.spmm(D, s_all)
            D = torch.spmm(D.transpose(0, 1), s_all).transpose(0, 1)
            loss_c = - (torch.trace(adj) / torch.trace(D))
            loss_o = torch.norm(s_all.transpose(0, 1) @ s_all / torch.norm(s_all.transpose(0, 1) @ s_all) - torch.eye(self.n_clusters[-1]).cuda() / (self.n_clusters[-1] ** (1/2)))
            # return x, adj, labels, syn_mask, 0, 0
            return x, adj, labels, syn_mask, loss_c, loss_o

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_syn, labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train():
        model.train()
        optimizer.zero_grad()
        _, mc_loss, o_loss = model(data.x, data.edge_index, data.edge_weight)
        loss = mc_loss + o_loss
        loss.backward()
        optimizer.step()
        return loss.item()


    @torch.no_grad()
    def test():
        model.eval()
        clust, _, _ = model(data.x, data.edge_index, data.edge_weight)
        return NMI(clust.max(1)[1].cpu(), data.y.cpu())

    def distance_wb(gwr, gws):
        shape = gwr.shape

        # TODO: output node!!!!
        if len(gwr.shape) == 2:
            gwr = gwr.T
            gws = gws.T

        if len(shape) == 4: # conv, out*in*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
            gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
        elif len(shape) == 3:  # layernorm, C*h*w
            gwr = gwr.reshape(shape[0], shape[1] * shape[2])
            gws = gws.reshape(shape[0], shape[1] * shape[2])
        elif len(shape) == 2: # linear, out*in
            tmp = 'do nothing'
        elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
            gwr = gwr.reshape(1, shape[0])
            gws = gws.reshape(1, shape[0])
            return 0

        dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
        dis = dis_weight
        return dis

    def match_loss(gw_syn, gw_real, dis_metric, device):
        dis = torch.tensor(0.0).to(device)

        if dis_metric == 'ours':

            for ig in range(len(gw_real)):
                gwr = gw_real[ig]
                gws = gw_syn[ig]
                dis += distance_wb(gwr, gws)

        elif dis_metric == 'mse':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

        elif dis_metric == 'cos':
            gw_real_vec = []
            gw_syn_vec = []
            for ig in range(len(gw_real)):
                gw_real_vec.append(gw_real[ig].reshape((-1)))
                gw_syn_vec.append(gw_syn[ig].reshape((-1)))
            gw_real_vec = torch.cat(gw_real_vec, dim=0)
            gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
            dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

        else:
            exit('DC error: unknown distance function')

        return dis

    SGC_layer = 2
    SGC_lr = 1e-2
    SGC_hidden = args.hid
    SGC_dropout = args.sgc_drop
    out_loop = args.out_loop
    inner_loop = args.inner_loop
    lambda_mc = 0
    lambda_o = 1    
    for epoch in range(args.epochs):
        # print(epoch)
        if args.dataset != 'ogbn-arxiv':
            gnn = SGC(nfeat=dataset.num_features, nhid=SGC_hidden,
                                        nclass=dataset.num_classes,
                                        dropout=SGC_dropout,
                                        nlayers=SGC_layer, with_bn=False,
                                        device='cuda').cuda()
        else:
            gnn = SGC1(nfeat=dataset.num_features, nhid=SGC_hidden,
                                    dropout=0.0, with_bn=False,
                                    weight_decay=0e-4, nlayers=2,
                                    nclass=dataset.num_classes,
                                    device='cuda').cuda()
        gnn.initialize()
        gnn_optimizer = torch.optim.Adam(gnn.parameters(), lr=SGC_lr)
        gnn.train()
        for oi in range(out_loop):
            x_syn, adj_syn, label_syn, syn_mask, loss_c, loss_o = model(sub_data.x, adj_full_norm)
            adj_syn = normalize_adj_tensor(adj_syn, sparse=False)
            syn_idx = {}
            num_class_dict = [0] * dataset.num_classes
            for c in range(dataset.num_classes):
                syn_idx[c] = ((label_syn == c) & syn_mask).nonzero().squeeze()
                num_class_dict[c] = syn_idx[c].numel()
            # loss = lambda_mc * mc_loss + lambda_o * o_loss
            loss = torch.tensor(0.0).to('cuda')
            for c in range(dataset.num_classes):
                syn_idx_c = syn_idx[c]
                if num_class_dict[c] == 0 or len(class_to_idx[c].shape) == 0:
                    continue
                output_real = gnn(sub_data.x, adj_full_norm)
                if len(class_to_idx[c]) > 1:
                    loss_real = F.nll_loss(output_real[class_to_idx[c]], sub_data.y[class_to_idx[c]])
                else:
                    loss_real = F.nll_loss(output_real[class_to_idx[c]].reshape(1, dataset.num_classes),
                     sub_data.y[class_to_idx[c]].reshape(1))
                gw_real = torch.autograd.grad(loss_real, gnn.parameters())
                gw_real = list((_.detach().clone() for _ in gw_real))
                output_syn = gnn(x_syn, adj_syn)
                if num_class_dict[c] > 1:
                    loss_syn = F.nll_loss(output_syn[syn_idx_c], label_syn[syn_idx_c])
                else:
                    loss_syn = F.nll_loss(output_syn[syn_idx_c].reshape(1, dataset.num_classes),
                    label_syn[syn_idx_c].reshape(1))
                gw_syn = torch.autograd.grad(loss_syn, gnn.parameters(), create_graph=True)
                coeff = num_class_dict[c] / max(num_class_dict)
                loss += coeff  * match_loss(gw_syn, gw_real, 'mse', device='cuda')

            #outer(train sc)
            optimizer.zero_grad()
            if oi % 50 >= args.max_out_match:
                loss = loss_c + loss_o
            loss.backward()
            optimizer.step()
            #inner(train gnn)
            x_inner, adj_inner = x_syn.detach(), adj_syn.detach()
            for j in range(1, inner_loop+1):
                gnn_optimizer.zero_grad()
                output_inner = gnn(x_inner, adj_inner)
                loss_inner = F.nll_loss(output_inner[syn_mask], label_syn[syn_mask])
                loss_inner.backward()
                gnn_optimizer.step()
    x_syn_l.append(x_syn.detach())
    adj_syn_l.append(adj_syn.detach())
    label_syn_l.append(label_syn.detach())
    syn_mask_l.append(syn_mask.detach())
    torch.save(x_syn, 'save_fast/x_syn_' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
    torch.save(adj_syn, 'save_fast/adj_syn' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
    torch.save(label_syn, 'save_fast/label_syn' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')
    torch.save(syn_mask, 'save_fast/syn_mask' + args.dataset + 'cluster' + str(cluster) + str(coarsening_ratio) + '.pt')

for i in range(args.cluster):
    condense_cluster(i, args.gpu_id)
    # print(torch.cuda.max_memory_allocated(args.gpu_id))

model = GCN(nfeat=data.x.shape[1], nhid=1024, dropout=0.5, lr=1e-3,
            weight_decay=0, nlayers=2, with_bn=True,
            nclass=dataset.num_classes, device='cuda').to('cuda')
# path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ogbn-arxiv')
# dataset = PygNodePropPredDataset('ogbn-arxiv', path)
# data = dataset[0].cuda()
# data.edge_index = utils.to_undirected(data.edge_index)
# data.y = data.y.squeeze(dim=-1)
# split_idx = dataset.get_idx_split()
# data.train_mask = torch.BoolTensor([False] * data.num_nodes).cuda()
# data.train_mask[split_idx["train"]] = torch.BoolTensor([True] * len(split_idx["train"])).cuda()
# data.val_mask = torch.BoolTensor([False] * data.num_nodes).cuda()
# data.val_mask[split_idx["valid"]] = torch.BoolTensor([True] * len(split_idx["valid"])).cuda()
# data.test_mask = torch.BoolTensor([False] * data.num_nodes).cuda()
# data.test_mask[split_idx["test"]] = torch.BoolTensor([True] * len(split_idx["test"])).cuda()
data = data.cuda()
will_merge = True
if will_merge:
    x_syn_l = torch.cat(x_syn_l, dim=0).cuda().detach()
    label_syn_l = torch.cat(label_syn_l).cuda().detach()
    syn_mask_l = torch.cat(syn_mask_l).cuda().detach()
    adj_syn_l = torch.block_diag(*adj_syn_l).cuda().detach()
model.fit_with_val(x_syn_l, adj_syn_l, label_syn_l, syn_mask_l, data,
                train_iters=800, normalize=False, verbose=False)

model.eval()
labels_test = data.y[data.test_mask]

labels_train = data.y[data.train_mask]
adj_full = torch.sparse_coo_tensor(data.edge_index,
                    torch.ones_like(data.edge_index[0]).float(), (data.num_nodes, data.num_nodes))
edge_index, edge_weight = gcn_norm(  
                    data.edge_index, data.edge_weight, data.num_nodes,
                    add_self_loops=True, dtype=data.x.dtype)
adj_full_norm = torch.sparse_coo_tensor(edge_index,
            edge_weight, (data.num_nodes, data.num_nodes))
output = model.predict(data.x, adj_full_norm)
loss_train = F.nll_loss(output[data.train_mask], labels_train)
acc_train = accuracy(output[data.train_mask], labels_train)
# print('train acc:', acc_train.item())
# Full graph
output = model.predict(data.x, adj_full_norm)
loss_test = F.nll_loss(output[data.test_mask], labels_test)
acc_test = accuracy(output[data.test_mask], labels_test)
print("test_acc:", acc_test.item())