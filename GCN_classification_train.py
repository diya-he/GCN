from __future__ import division
from __future__ import print_function
import time
import argparse
from tensorboardX import SummaryWriter
from models import *
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
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
parser.add_argument("--normalization", default="FirstOrderGCN",
                    help="The normalization on the adj matrix.")
parser.add_argument('--dataset', default="cora", help="The data set")           #citeseer  cora
parser.add_argument('--datapath', default="./data/", help="The data path.")
parser.add_argument("--no_tensorboard", default=False, help="Disable writing logs to tensorboard")
parser.add_argument('--lradjust', action='store_true',
                    default=True, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
OUTPUT_PATH = r'./classification/'
test_flag = True
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

# random seed setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# define the training function.
def train(model, train_adj, train_fea, idx_train, val_adj=None, val_fea=None):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(train_fea, train_adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    train_t = time.time() - t
    val_t = time.time()
    if not args.fastmode:
        model.eval()
        output = model(val_fea, val_adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
    acc_val = accuracy(output[idx_val], labels[idx_val]).item()
    if args.lradjust:
        scheduler.step()
    val_t = time.time() - val_t
    return (loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer), train_t, val_t)


def test(model, test_adj, test_fea):
    model.eval()
    output = model(test_fea, test_adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    auc_test = roc_auc_compute_fn(output[idx_test], labels[idx_test])
    return (loss_test.item(), acc_test.item())

DropEdge_sampler = DropEdge(args.dataset, args.datapath)
# get labels and indexes
labels, idx_train, idx_val, idx_test = DropEdge_sampler.get_label_and_idxes(args.cuda)
nfeat = DropEdge_sampler.nfeat
nclass = DropEdge_sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))
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
for i_case, kwargs in enumerate(test_cases):
    args.nbaseblocklayer = kwargs['num_layers']
    args.withloop = kwargs['add_self_loops']
    args.withbn = kwargs['add_bn']
    args.use_pairnorm = kwargs['use_pairnorm']
    args.sampling_percent = kwargs['drop_edge']
    args.activation = kwargs['activation']

    model = MUTILAYERGCN(nfeat=nfeat,
                         nhid=args.hidden,
                         nclass=nclass,
                         dropout=args.dropout,
                         self_loops=args.withloop,
                         num_layers=args.nbaseblocklayer,
                         norm_mode=args.withbn,
                         use_pairnorm=args.use_pairnorm,  # 'None', 'PN', 'PN-SI', 'PN-SCS'
                         activation=args.activation)  # relu  linear
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # lr=1e-3  注意：在2.17日15.20之前的数据的学习率均固定为1e-3
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lradjust:
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.618)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.5)

    # convert to cuda
    model.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    if args.no_tensorboard is False:
        tb_writer = SummaryWriter(
            comment=f"-dataset_{args.dataset}"
        )

    # Train model
    t_total = time.time()
    loss_train = np.zeros((args.epochs,))
    acc_train = np.zeros((args.epochs,))
    loss_val = np.zeros((args.epochs,))
    acc_val = np.zeros((args.epochs,))
    lr = np.zeros((args.epochs,))
    sampling_t = 0
    best_acc = 0
    best_loss = 1e10
    for epoch in range(args.epochs):
        input_idx_train = idx_train
        sampling_t = time.time()
        (train_adj, train_fea) = DropEdge_sampler.randomedge_sampler(percent=args.sampling_percent,
                                                            normalization=args.normalization,
                                                            cuda=args.cuda)
        train_adj = train_adj.cuda()
        sampling_t = time.time() - sampling_t
        (val_adj, val_fea) = DropEdge_sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
        val_adj = val_adj.cuda()
        outputs = train(model, train_adj, train_fea, input_idx_train, val_adj, val_fea)
        if args.no_tensorboard is False:
            tb_writer.add_scalars('Loss', {'train': outputs[0], 'val': outputs[2]}, epoch)
            tb_writer.add_scalars('Accuracy', {'train': outputs[1], 'val': outputs[3]}, epoch)
            tb_writer.add_scalar('lr', outputs[4], epoch)
            tb_writer.add_scalars('Time', {'train': outputs[5], 'val': outputs[6]}, epoch)
        loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch], lr[epoch] = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
        print("train_loss:"+ str(outputs[0])+ " "+"train_acc:" + str(outputs[1])+" "+"lr:"+str(outputs[4])+" "+"Time:"+str(outputs[5]))
        val_acc = outputs[3]
        # save model
        if test_flag:
            if best_acc < val_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), OUTPUT_PATH+'checkpoint-best-acc.pkl')
        # torch.save(model.state_dict(), OUTPUT_PATH + 'checkpoint-best-acc.pkl')
    # Testing
    if test_flag:
        model.load_state_dict(torch.load(OUTPUT_PATH + 'checkpoint-best-acc.pkl'))
        (test_adj, test_fea) = DropEdge_sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
        test_adj = test_adj.cuda()
        (loss_test, acc_test) = test(model, test_adj, test_fea)
        print("%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f" % (
            i_case, lr[-1], loss_train[-1], loss_val[-1], loss_test, acc_train[-1], acc_val[-1], acc_test))
    # else:
        kwargs['best_acc'] = max(acc_val)
        args.nbaseblocklayer = kwargs['num_layers']
        args.withloop = kwargs['add_self_loops']
        args.withbn = kwargs['add_bn']
        args.sampling_percent = kwargs['drop_edge']
        args.use_pairnorm = kwargs['use_pairnorm']
        file_path = OUTPUT_PATH
        # nameloss = 'case%sLr%floss_layer%iselfloop%iwithbn%iuse_pairnorm%sdrop_edge%factivation%s' % (
        #     i_case, lr[-1], args.nbaseblocklayer, args.withloop, args.withbn, args.use_pairnorm, args.sampling_percent, args.activation)
        # nameacc = 'case%sLr%facc_layer%iselfloop%iwithbn%iuse_pairnorm%sdrop_edge%factivation%s' % (
        #     i_case, lr[-1], args.nbaseblocklayer, args.withloop, args.withbn, args.use_pairnorm, args.sampling_percent, args.activation)
        plt.plot(list(range(args.epochs)), loss_train, label='loss_train')
        plt.plot(list(range(args.epochs)), loss_val, label='loss_val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('train loss VS val loss')
        plt.legend()
        plt.savefig(file_path +  "loss.jpg")
        # plt.show()

        plt.clf() 
        plt.plot(list(range(args.epochs)), acc_train, label='acc_train')
        plt.plot(list(range(args.epochs)), acc_val, label='acc_val')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('train acc VS val acc')
        plt.legend()
        plt.savefig(file_path + "acc.jpg")
        # plt.show()

        # pkl_name1 = 'case%iLr%flayer%iselfloop%iwithbn%iuse_pairnorm%sdrop_edge%factivation%sloss_train.pkl' % (
        #     i_case, lr[-1], args.nbaseblocklayer, args.withloop, args.withbn, args.use_pairnorm, args.sampling_percent, args.activation)
        # with open(pkl_name1, 'wb') as f:
        #     pickle.dump(loss_train, f)
        # pkl_name2 = 'case%iLr%flayer%iselfloop%iwithbn%iuse_pairnorm%sdrop_edge%factivation%sloss_val.pkl' % (
        #     i_case, lr[-1], args.nbaseblocklayer, args.withloop, args.withbn, args.use_pairnorm, args.sampling_percent, args.activation)
        # with open(pkl_name2, 'wb') as f:
        #     pickle.dump(loss_val, f)
        # pkl_name3 = 'case%iLr%flayer%iselfloop%iwithbn%iuse_pairnorm%sdrop_edge%factivation%sacc_train.pkl' % (
        #     i_case, lr[-1], args.nbaseblocklayer, args.withloop, args.withbn, args.use_pairnorm, args.sampling_percent, args.activation)
        # with open(pkl_name3, 'wb') as f:
        #     pickle.dump(acc_train, f)
        # pkl_name4 = 'case%iLr%flayer%iselfloop%iwithbn%iuse_pairnorm%sdrop_edge%factivation%sacc_val.pkl' % (
        #     i_case, lr[-1], args.nbaseblocklayer, args.withloop, args.withbn, args.use_pairnorm, args.sampling_percent, args.activation)
        # with open(pkl_name4, 'wb') as f:
        #     pickle.dump(acc_val, f)
if not test_flag:
    pd.DataFrame(test_cases).to_csv(f'{args.dataset}-Result.csv')






