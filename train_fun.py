import time
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import torch.optim as optim
import torch
from torch.autograd import Variable
import numpy as np
import networkx as nx
from torchsampler import ImbalancedDatasetSampler
import torch.nn.functional as F

import config
import graph_fun
import loss_fun
import utils_fun
import net_class

torch.cuda.set_device(config.cuda_num)
os.environ['CUDA_VISIBLE_DIVICES'] = str(config.cuda_num)
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
net = net_class.ResNet18()
if cuda:
    net = net.cuda()

Source_dataset = datasets.ImageFolder(config.source_dataset_path, transform=config.trainTransform)
train_dataloader = DataLoader(Source_dataset, int(config.source_batch_size*(6/10)),
                              sampler=ImbalancedDatasetSampler(Source_dataset))  # train1
train_dataloader1 = DataLoader(Source_dataset, 1, sampler=ImbalancedDatasetSampler(Source_dataset))  # train1
source_dataloader = DataLoader(Source_dataset, config.GCN_source_batch_size, shuffle=False)  # train2

Target_dataset = datasets.ImageFolder(config.target_dataset_path, transform=config.trainTransform)
target_dataloader = DataLoader(Target_dataset, config.target_batch_size, shuffle=True)  # train1
test_target_dataloader = DataLoader(Target_dataset, config.target_batch_size, shuffle=False)  # train2

mmd_loss = loss_fun.MMD_loss()
con_loss = loss_fun.ContrastiveLoss()
# optimizer = optim.SGD(net.parameters(), lr=config.lr_init, momentum=0.9, dampening=0.1)
optimizer = optim.Adam(net.parameters(), lr=config.features_lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


def generate_batch(epoch, size=int(config.source_batch_size * (4/10))):
    sh_label = epoch % config.class_num
    source_train_iter = iter(train_dataloader)
    source_train_iter1 = iter(train_dataloader1)
    target_train_iter = iter(target_dataloader)
    input_1, label_1 = next(source_train_iter)
    input_2, label_2 = next(source_train_iter)
    input_3, label_3 = next(target_train_iter)
    input_11_l = []
    input_12_l = []
    label_11_l = []
    label_12_l = []

    while 1:
        input_11, label_11 = next(source_train_iter1)
        input_12, label_12 = next(source_train_iter1)
        if label_11.numpy()[0] == sh_label:
            input_11_l.append(input_11)
            label_11_l.append(label_11)
        if label_12.numpy()[0] == sh_label:
            input_12_l.append(input_12)
            label_12_l.append(label_12)
        if len(label_12_l) >= size and len(label_11_l) >= size:
            break
    label_11_l = label_11_l[:size]
    label_12_l = label_12_l[:size]
    input_11_l = input_11_l[:size]
    input_12_l = input_12_l[:size]

    label_11_new = torch.cat(label_11_l, 0)
    label_12_new = torch.cat(label_12_l, 0)
    input_11_new = torch.cat(input_11_l, 0)
    input_12_new = torch.cat(input_12_l, 0)

    label_1_new = torch.cat((label_1, label_11_new), 0)
    label_2_new = torch.cat((label_2, label_12_new), 0)
    input_1_new = torch.cat((input_1, input_11_new), 0)
    input_2_new = torch.cat((input_2, input_12_new), 0)

    label = (label_1_new.numpy() == label_2_new.numpy()).astype('float32')
    return input_1_new, input_2_new, input_3, label


def train1(bal=1, mode=1):
    best_acc = 0
    list_mmd = []
    list_con = []
    list_sum = []
    list_acc = []
    for epoch in range(config.iter_num):
        since = time.time()
        input_1, input_2, input_3, out = generate_batch(epoch)
        X_1 = Variable(torch.Tensor(input_1).float()).cuda()
        X_2 = Variable(torch.Tensor(input_2).float()).cuda()
        X_3 = Variable(torch.Tensor(input_3).float()).cuda()
        Y = Variable(torch.Tensor(out).float()).cuda()
        optimizer.zero_grad()
        out_1, out_2, out_3 = net(X_1, X_2, X_3)
        loss_con = con_loss(out_1, out_2, Y)
        loss_mmd = mmd_loss(out_3, out_2)
        loss_sum = loss_con + bal * loss_mmd
        list_con.append(loss_con.item())
        list_mmd.append(loss_mmd.item())
        list_sum.append(loss_sum.item())
        loss_sum.backward()
        optimizer.step()
        print("Iter=", epoch, 'bal=', bal, "loss_con=", loss_con.item(), 'loss_mmd=',
              loss_mmd.item())
        if epoch % config.test_frequency == 0:
            the_time = time.time() - since
            torch.save(net.state_dict(), os.path.join(config.model_dir, config.model_name + '-' + str(mode)
                                                      , config.model_file))
            _, labels_, _ = graph_fun.K_means(test_target_dataloader, net)
            acc, labels_pred = utils_fun.test_acc(labels_)
            list_acc.append(acc)
            utils_fun.save_txt_file(os.path.join(config.model_dir, config.model_name + '-' + str(mode),
                                                 config.acc_file), str(acc), 'a')
            utils_fun.save_txt_file(os.path.join(config.model_dir, config.model_name + '-' + str(mode),
                                                 config.con_file), str(loss_con.item()), 'a')
            utils_fun.save_txt_file(os.path.join(config.model_dir, config.model_name + '-' + str(mode),
                                                 config.mmd_file), str(loss_mmd.item()), 'a')
            utils_fun.save_txt_file(os.path.join(config.model_dir, config.model_name + '-' + str(mode),
                                                 config.loss_file), str(loss_sum.item()), 'a')
            np.save(os.path.join(config.model_dir, config.model_name + '-' + str(mode), 'labels_pred.npy'), labels_pred)
            if acc >= best_acc:
                best_acc = acc
                log_dir = os.path.join(config.model_dir, config.model_name + '-' + str(mode),
                                       str(epoch) + '_' + str(int(acc * 100)))
                if os.path.exists(log_dir) is not True:
                    os.makedirs(log_dir)
                utils_fun.save_txt_file(os.path.join(log_dir, config.time_file), str(the_time / config.test_frequency)
                                        , 'w')
                utils_fun.save_txt_file(os.path.join(log_dir, config.bal_file), str(bal), 'w')
                utils_fun.save_txt_files(os.path.join(log_dir, config.acc_file), list_acc)
                utils_fun.save_txt_files(os.path.join(log_dir, config.con_file), list_con)
                utils_fun.save_txt_files(os.path.join(log_dir, config.mmd_file), list_mmd)
                utils_fun.save_txt_files(os.path.join(log_dir, config.loss_file), list_sum)
                np.save(os.path.join(log_dir, 'labels_pred.npy'), labels_pred)
                torch.save(net.state_dict(), os.path.join(log_dir, config.model_file))
            print("Iter=", epoch, 'time=', the_time, 'bal=', bal, "loss_con=", loss_con.item(), 'loss_mmd=',
                  loss_mmd.item(), 'acc=', acc)


def train2():
    learning_rate = config.GCN_lr
    source_center_features = graph_fun.K_clus(source_dataloader, net)
    cluster_centers_, labels_, features = graph_fun.K_means(test_target_dataloader, net)
    acc, _ = utils_fun.test_acc(labels_)
    print('Feature extraction acc:', acc)
    graph_fun.graph(cluster_centers_, labels_, features, source_center_features)
    best_acc = 0
    list_loss = []
    list_acc = []
    for epoch in range(config.GCN_iter_num):
        since = time.time()
        sum_center = torch.zeros(config.features_dim_num)
        if cuda:
            sum_center = sum_center.cuda()
        GCN_features = []
        GCN_source_features = []
        for i in range(0, config.class_num):
            g = nx.read_gpickle(os.path.join(config.model_dir, 'GCN-1',
                                             'Target_Kgraph/graph_feature_{}.gpickle'.format(i + 1)))
            x = np.load(os.path.join(config.model_dir, 'GCN-1', 'Target_Kgraph/X_{}_fetures.npy'.format(i + 1)))
            x = np.mat(x)
            A, D = graph_fun.preprocess(g)

            FloatTensor = torch.FloatTensor
            # Turn the input and output into FloatTensors for the Neural Network

            x = Variable(FloatTensor(x), requires_grad=False)
            A = torch.from_numpy(A)
            A = A.float()
            A = Variable(A, requires_grad=False)
            D = torch.from_numpy(D)
            D = D.float()
            D = Variable(D, requires_grad=False)
            if cuda:
                A, D = A.cuda(), D.cuda()
                x = x.cuda()
            # Create random tensor weights
            if os.path.exists(os.path.join(config.model_dir, 'GCN-1', 'W1.npy')) is not True:
                W1 = Variable(torch.randn(config.features_dim_num, config.features_dim_num).type(FloatTensor),
                              requires_grad=True)
                W2 = Variable(torch.randn(config.features_dim_num, config.features_dim_num).type(FloatTensor),
                              requires_grad=True)
                W3 = Variable(torch.randn(config.features_dim_num, config.features_dim_num).type(FloatTensor),
                              requires_grad=True)
            else:
                W1 = Variable(torch.tensor(np.load(os.path.join(config.model_dir, 'GCN-1', 'W1.npy'))).type(FloatTensor),
                              requires_grad=True)
                W2 = Variable(torch.tensor(np.load(os.path.join(config.model_dir, 'GCN-1', 'W2.npy'))).type(FloatTensor),
                              requires_grad=True)
                W3 = Variable(torch.tensor(np.load(os.path.join(config.model_dir, 'GCN-1', 'W3.npy'))).type(FloatTensor),
                              requires_grad=True)
            if cuda:
                W1, W2, W3 = W1.cuda(), W2.cuda(), W3.cuda()
            hidden_layer_1 = F.relu(D.mm(A).mm(D).mm(x).mm(W1))
            W1.retain_grad()
            hidden_layer_2 = F.relu(D.mm(A).mm(D).mm(hidden_layer_1).mm(W2))
            W2.retain_grad()
            y_pred = D.mm(A).mm(D).mm(hidden_layer_2).mm(W3)
            W3.retain_grad()
            GCN_features.append(y_pred.data.cpu().numpy()[1:])
            GCN_source_features.append(y_pred.data.cpu().numpy()[0])
            center = torch.mean(y_pred, dim=0)
            for i in y_pred:
                sum_center += torch.abs(center - i)
        loss_sum = sum_center.mean() / 2
        list_loss.append(loss_sum.item())
        loss_sum.backward()
        # Update weights using gradient descent
        if epoch % 20 == 0:
            learning_rate = learning_rate / 10
            print('learning_rate:', learning_rate)
        W1 -= learning_rate * W1.grad
        W2 -= learning_rate * W2.grad
        W3 -= learning_rate * W3.grad

        np.save(os.path.join(config.model_dir, 'GCN-1', 'W1.npy'), W1.data.cpu().numpy())
        np.save(os.path.join(config.model_dir, 'GCN-1', 'W2.npy'), W2.data.cpu().numpy())
        np.save(os.path.join(config.model_dir, 'GCN-1', 'W3.npy'), W3.data.cpu().numpy())
        np.save(os.path.join(config.model_dir, 'GCN-1', 'GCN_features.npy'), np.concatenate(GCN_features))
        np.save(os.path.join(config.model_dir, 'GCN-1', 'GCN_source_features.npy'), GCN_source_features)
        GCN_source_features_ = np.load(os.path.join(config.model_dir, 'GCN-1', 'GCN_source_features.npy'))
        utils_fun.save_txt_file(os.path.join(config.model_dir, 'GCN-1', config.loss_file), str(loss_sum.item()), 'a')
        utils_fun.save_txt_file(os.path.join(config.model_dir, 'GCN-1', config.acc_file), str(acc), 'a')

        if epoch % config.GCN_test_frequency == 0:
            the_time = time.time() - since
            cluster_centers_, labels_, GCN_features = graph_fun.K_means(test_target_dataloader, net)
            acc, labels_pred = utils_fun.test_acc(labels_)
            np.save(os.path.join(config.model_dir, 'GCN-1', 'labels_pred.npy'), labels_pred)
            list_acc.append(acc)
            if acc >= best_acc:
                best_acc = acc
                log_dir = os.path.join(config.model_dir, 'GCN-1', str(epoch) + str(int(acc * 100)))
                os.makedirs(log_dir)
                utils_fun.save_txt_files(os.path.join(log_dir, config.loss_file), list_loss)
                utils_fun.save_txt_files(os.path.join(log_dir, config.acc_file), list_acc)
                utils_fun.save_txt_file(os.path.join(log_dir, config.time_file),
                                        str(the_time / config.GCN_test_frequency), 'w')
                np.save(os.path.join(log_dir, 'GCN_features.npy'), np.concatenate(GCN_features))
                np.save(os.path.join(log_dir, 'GCN_source_features.npy'), GCN_source_features)
                np.save(os.path.join(log_dir, 'labels_pred'), labels_pred)
                graph_fun.graph(cluster_centers_, labels_, features, source_center_features, GCN_source_features=GCN_source_features_, graph_path=log_dir)
            else:
                graph_fun.graph(cluster_centers_, labels_, features, source_center_features, GCN_source_features=GCN_source_features_)
            print('epoch:', epoch, 'time:', the_time, 'K_loss:', loss_sum.item(), 'acc:', acc)





















