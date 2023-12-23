from __future__ import division
from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from models import *
from utils import *
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling
import warnings
warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.8,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument("--normalization", default="FirstOrderGCN",  # FirstOrderGCN
                    help="The normalization on the adj matrix.")
parser.add_argument('--dataset', default="cora", help="The data set")  # citeseer  cora
parser.add_argument('--datapath', default="./data", help="The data path.")
parser.add_argument('--lradjust', action='store_true',
                    default=True, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
OUTPUT_PATH = r'./predict/'
test_flag = True
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
# random seed setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)

@torch.no_grad()
def eval_link_predictor(model, data, adj_test):
    model.eval()
    z = model.encode(data.x, adj_test)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# define the training function.
def train(model, train_data, adj_train_nrom, val_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, adj_train_nrom)
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)
    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    val_auc = eval_link_predictor(model, val_data, adj_train_nrom)
    return loss, val_auc, get_lr(optimizer)

def test(model, test_data, adj_test_nrom):
    model.eval()
    test_auc = eval_link_predictor(model, test_data, adj_test_nrom)
    return test_auc
if not test_flag:
    test_cases = [
        # base
        {'num_layers':2, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        # test layer 1
        {'num_layers':3, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':4, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':8, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':16, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':32, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        # test self loops 6
        {'num_layers':2, 'add_self_loops':True, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':4, 'add_self_loops':True, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':8, 'add_self_loops':True, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':16, 'add_self_loops':True, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':32, 'add_self_loops':True, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        # test batch norm 11
        {'num_layers':2, 'add_self_loops':False, 'add_bn':True, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':4, 'add_self_loops':False, 'add_bn':True, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':8, 'add_self_loops':False, 'add_bn':True, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':16, 'add_self_loops':False, 'add_bn':True, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':32, 'add_self_loops':False, 'add_bn':True, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'linear'},
        # test use_pairnorm 16
        {'num_layers':2, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'PN', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':4, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'PN', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':8, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'PN', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':16, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'PN', 'drop_edge':1.0, 'activation':'linear'},
        {'num_layers':32, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'PN', 'drop_edge':1.0, 'activation':'linear'},
        # test drop_edge 21
        {'num_layers':2, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.8, 'activation':'linear'},
        {'num_layers':4, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.8, 'activation':'linear'},
        {'num_layers':8, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.8, 'activation':'linear'},
        {'num_layers':16, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.8, 'activation':'linear'},
        {'num_layers':32, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.8, 'activation':'linear'},
        {'num_layers':2, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.6, 'activation':'linear'},
        {'num_layers':4, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.6, 'activation':'linear'},
        {'num_layers':8, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.6, 'activation':'linear'},
        {'num_layers':16, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.6, 'activation':'linear'},
        {'num_layers':32, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.6, 'activation':'linear'},
        {'num_layers':2, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.4, 'activation':'linear'},
        {'num_layers':4, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.4, 'activation':'linear'},
        {'num_layers':8, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.4, 'activation':'linear'},
        {'num_layers':16, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.4, 'activation':'linear'},
        {'num_layers':32, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':0.4, 'activation':'linear'},
        # test activation 36
        {'num_layers':2, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'tanh'},
        {'num_layers':4, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'tanh'},
        {'num_layers':8, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'tanh'},
        {'num_layers':16, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'tanh'},
        {'num_layers':32, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'tanh'},
        {'num_layers':2, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'relu'},
        {'num_layers':4, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'relu'},
        {'num_layers':8, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'relu'},
        {'num_layers':16, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'relu'},
        {'num_layers':32, 'add_self_loops':False, 'add_bn':False, 'use_pairnorm':'None', 'drop_edge':1.0, 'activation':'relu'},
    ]
else:
    test_cases = [
        {'num_layers':8, 'add_self_loops':True, 'add_bn':True, 'use_pairnorm':'PN', 'drop_edge':1.0, 'activation':'linear'},
    ]
dataset = Planetoid(root=args.datapath, name=args.dataset)
graph = dataset[0]
del graph.train_mask
del graph.val_mask
del graph.test_mask
split = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0,
)
train_data, val_data, test_data = split(graph)
adj_train = sp.csr_matrix(
        (np.ones(train_data.num_edges), (train_data.edge_index[0, :], train_data.edge_index[1, :])),
        shape=[train_data.num_nodes, train_data.num_nodes])
adj_train_nrom = adj_normalize(adj_train)
adj_train_nrom = sparse_mx_to_torch_sparse_tensor(adj_train_nrom)

adj_test = sp.csr_matrix((np.ones(test_data.num_edges), (test_data.edge_index[0, :], test_data.edge_index[1, :])),
                            shape=[test_data.num_nodes, test_data.num_nodes])
adj_test_nrom = adj_normalize(adj_test)
adj_test_nrom = sparse_mx_to_torch_sparse_tensor(adj_test_nrom)
# convert to cuda
train_data = train_data.cuda()
val_data = val_data.cuda()
test_data = test_data.cuda()
adj_train_nrom = adj_train_nrom.cuda()
adj_test_nrom = adj_test_nrom.cuda()
for i_case, kwargs in enumerate(test_cases):
    args.nbaseblocklayer = kwargs['num_layers']
    args.withloop = kwargs['add_self_loops']
    args.withbn = kwargs['add_bn']
    args.use_pairnorm = kwargs['use_pairnorm']
    args.sampling_percent = kwargs['drop_edge']
    args.activation = kwargs['activation']
    nfeat = dataset.num_node_features
    nclass = dataset.num_classes
    model = LinkNet(nfeat=nfeat,
                    nhid=args.hidden,
                    nclass=nclass,
                    dropout=args.dropout,
                    self_loops=args.withloop,
                    num_layers=args.nbaseblocklayer,
                    norm_mode=args.withbn,
                    use_pairnorm=args.use_pairnorm,  # 'None', 'PN', 'PN-SI', 'PN-SCS'
                    activation=args.activation)  # relu  linear
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # lr=1e-3  注意：在2.17日15.20之前的数据的学习率均固定为1e-3
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lradjust:
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.618)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.5)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train model
    loss_train = np.zeros((args.epochs,))
    auc_val = np.zeros((args.epochs,))
    lr = np.zeros((args.epochs,))
    best_auc =0
    for epoch in range(args.epochs):
        loss, val_auc, lr_get = train(model, train_data, adj_train_nrom, val_data)
        print("loss:" + str(loss.item()) + " "+"val_auc:" + str(val_auc.item())+" "+"lr:"+str(lr_get))
        loss_train[epoch], auc_val[epoch], lr[epoch] = loss, val_auc, lr_get
        if test_flag:
            if best_auc < val_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), OUTPUT_PATH + 'checkpoint-best-auc.pkl')
    # Testing
    if test_flag:
        model.load_state_dict(torch.load(OUTPUT_PATH + 'checkpoint-best-auc.pkl'))
        test_auc = test(model, test_data, adj_test_nrom)
        print("%i\t%.6f\t%.6f\t%.6f\t%.6f" % (
            i_case, lr[-1], loss_train[-1], auc_val[-1], test_auc))
    # else:
        kwargs['best_auc'] = max(auc_val)
        args.nbaseblocklayer = kwargs['num_layers']
        args.withloop = kwargs['add_self_loops']
        args.withbn = kwargs['add_bn']
        args.sampling_percent = kwargs['drop_edge']
        args.use_pairnorm = kwargs['use_pairnorm']
        file_path=OUTPUT_PATH
        # nameloss = 'case%sLr%floss_layer%iselfloop%iwithbn%iuse_pairnorm%sdrop_edge%factivation%s' % (
        #     i_case, lr[-1], args.nbaseblocklayer, args.withloop, args.withbn, args.use_pairnorm, args.sampling_percent,
        #     args.activation)
        # nameacc = 'case%sLr%facc_layer%iselfloop%iwithbn%iuse_pairnorm%sdrop_edge%factivation%s' % (
        #     i_case, lr[-1], args.nbaseblocklayer, args.withloop, args.withbn, args.use_pairnorm, args.sampling_percent,
        #     args.activation)
        plt.plot(list(range(args.epochs)), loss_train, label='loss_train')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('train loss VS val loss')
        plt.legend()
        plt.savefig(file_path + "loss.jpg")
        # plt.show()
        plt.clf()
        plt.plot(list(range(args.epochs)), auc_val, label='auc_val')
        plt.xlabel('epoch')
        plt.ylabel('auc')
        plt.title('train auc VS val auc')
        plt.legend()
        plt.savefig(file_path + "auc.jpg")
        plt.show()

        # pkl_name1 = 'task%icase%iLr%flayer%iselfloop%iwithbn%iuse_pairnorm%sdrop_edge%factivation%sloss_train.pkl' % (
        #     task, i_case, lr[-1], args.nbaseblocklayer, args.withloop, args.withbn, args.use_pairnorm, args.sampling_percent,
        #     args.activation)
        # with open(pkl_name1, 'wb') as f:
        #     pickle.dump(loss_train, f)
        # pkl_name2 = 'task%icase%iLr%flayer%iselfloop%iwithbn%iuse_pairnorm%sdrop_edge%factivation%sacc_val.pkl' % (
        #     task, i_case, lr[-1], args.nbaseblocklayer, args.withloop, args.withbn, args.use_pairnorm, args.sampling_percent,
        #     args.activation)
        # with open(pkl_name2, 'wb') as f:
        #     pickle.dump(auc_val, f)
        # pkl_name3 = 'task%icase%iLr%flayer%iselfloop%iwithbn%iuse_pairnorm%sdrop_edge%factivation%sacc_val.pkl' % (
        #     task, i_case, lr[-1], args.nbaseblocklayer, args.withloop, args.withbn, args.use_pairnorm, args.sampling_percent,
        #     args.activation)
        # with open(pkl_name3, 'wb') as f:
        #     pickle.dump(lr, f)
# if not test_flag:
#     if task == 1:
#         pd.DataFrame(test_cases).to_csv(f'{args.dataset1}-Result1.csv')
#     else:
#         pd.DataFrame(test_cases).to_csv(f'{args.dataset2}-Result11.csv')
