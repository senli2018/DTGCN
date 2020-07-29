import os
import config
import numpy as np
import torch
import networkx as nx
from sklearn.cluster import KMeans
import joblib
from torch.autograd import Variable


def distance(features_1, features_2):
    dist = np.sqrt(np.sum(np.square(features_1 - features_2)))
    return dist


def K_means(dataloader, net):
    if os.path.exists(os.path.join(config.model_dir, 'GCN-1', 'GCN_features.npy')) is not True:
        F = np.load('Model_and_Log/features.npy', allow_pickle=True)
        # for i, data in enumerate(dataloader):
        #     inputs, label = data
        #     inputs = Variable(torch.Tensor(inputs).float()).cuda()
        #     net.load_state_dict(torch.load(config.kmeans_model),
        #                         strict=False)
        #     net.eval()
        #     feature, _, _ = net(inputs, inputs, inputs)
        #     print('test_feature shape:', feature.cpu().detach().numpy().shape)
        #     for j in feature:
        #         F.append(j.cpu().detach().numpy())
        #     print("extract the {} feature".format(i * config.target_batch_size))
        # np.save(os.path.join(config.model_dir, 'features.npy'), F)
        # F = np.concatenate(F)
    else:
        F = np.load(os.path.join(config.model_dir, 'GCN-1', 'GCN_features.npy'), allow_pickle=True)
    kmeans = KMeans(n_clusters=config.class_num, random_state=0).fit(F)
    joblib.dump(kmeans, 'km.pkl')
    return kmeans.cluster_centers_, kmeans.labels_, F


def K_clus(dataloader, net):
    F = []
    if os.path.exists('Model_and_Log/source_center_features.npy') is not True:
        for i, data in enumerate(dataloader):
            inputs, label = data
            inputs = Variable(torch.Tensor(inputs).float()).cuda()
            net.load_state_dict(torch.load(config.kclus_model), strict=False)
            net.eval()
            feature, _, _ = net(inputs, inputs, inputs)
            print('test_feature shape:', feature.cpu().detach().numpy().shape)
            for j in feature:
                F.append(j.cpu().detach().numpy())
            print("extract the {} feature".format(i * config.GCN_source_batch_size))
        np.save(os.path.join(config.model_dir, 'source_features.npy'), F)
        source_features = np.load(os.path.join(config.model_dir, 'source_features.npy'), allow_pickle=True)
        F1 = []
        F2 = []
        F3 = []
        F4 = []
        F5 = []
        F6 = []
        class1 = config.train_class1_num
        class2 = class1 + config.train_class2_num
        class3 = class2 + config.train_class3_num
        class4 = class3 + config.train_class4_num
        class5 = class4 + config.train_class5_num
        class6 = class5 + config.train_class6_num
        F = []
        for i in range(class6):
            if i <= class1:
                F1.append(source_features[i])
            elif class1 < i <= class2:
                F2.append(source_features[i])
            elif class2 < i <= class3:
                F3.append(source_features[i])
            elif class3 < i <= class4:
                F4.append(source_features[i])
            elif class4 < i <= class5:
                F5.append(source_features[i])
            elif class5 < i <= class6:
                F6.append(source_features[i])
        f1 = sum(F1) / len(F1)
        f2 = sum(F2) / len(F2)
        f3 = sum(F3) / len(F3)
        f4 = sum(F4) / len(F4)
        f5 = sum(F5) / len(F5)
        f6 = sum(F6) / len(F6)

        F.append(f1)
        F.append(f2)
        F.append(f3)
        F.append(f4)
        F.append(f5)
        F.append(f6)
        np.save(os.path.join(config.model_dir, 'source_center_features.npy'), F)
        F = np.concatenate(F)
    else:
        F = np.load(os.path.join(config.model_dir, 'source_center_features.npy'), allow_pickle=True)
    return F


def graph(cluster_centers_, labels_, features, source_center_features, GCN_source_features=None, graph_path=None):
    if GCN_source_features is None:
        k = source_center_features
    else:
        k = GCN_source_features
        k1 = source_center_features
    k_cluster = cluster_centers_
    labelPred = labels_
    F = features
    X = [6, 6, 6, 6, 6, 6]
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X5 = []
    X6 = []
    Y = []
    Y1 = []
    gn1 = []
    ge1 = []
    gn2 = []
    ge2 = []
    gn3 = []
    ge3 = []
    gn4 = []
    ge4 = []
    gn5 = []
    ge5 = []
    gn6 = []
    ge6 = []
    for x in range(0, len(k)):
        dis_list = []
        for j in range(0, len(k_cluster)):
            d = distance(k[x], k_cluster[j])
            dis_list.append(d)
        Y.append(dis_list)
        dis_inx = np.array(dis_list)
        dis_inx = dis_inx.argsort()
        Y1.append(dis_inx)
    for x in range(0, len(Y)):
        if X[x] != 6:
            pass
        elif x == 5:
            X[x] = 21 - sum(X)
            break
        else:
            flag1 = 0
            while (flag1 == 0):
                con = []
                ix = []
                flag = 0
                for i in range(x + 1, 6):
                    # print(x)
                    while (Y1[x][flag] in X):
                        flag += 1
                    if Y[x][flag] not in con:
                        con.append(Y[x][flag])
                    if x not in ix:
                        ix.append(x)
                    if Y1[x][flag] == Y1[i][flag]:
                        if X[i] == 6:
                            con.append(Y[i][flag])
                            ix.append(i)
                        else:
                            pass
                con_sort = np.array(con)
                con_sort = con_sort.argsort()
                X[ix[con_sort[0]]] = Y1[x][flag]
                if ix[con_sort[0]] == x:
                    flag1 = 1

    G1 = nx.Graph()
    G2 = nx.Graph()
    G3 = nx.Graph()
    G4 = nx.Graph()
    G5 = nx.Graph()
    G6 = nx.Graph()
    G1.add_node('a')
    G2.add_node('b')
    G3.add_node('c')
    G4.add_node('d')
    G5.add_node('e')
    G6.add_node('f')
    np.save('X.npy', X)
    if GCN_source_features is None:
        X1.append(k[0])
        X2.append(k[1])
        X3.append(k[2])
        X4.append(k[3])
        X5.append(k[4])
        X6.append(k[5])
    else:
        X1.append(k1[0])
        X2.append(k1[1])
        X3.append(k1[2])
        X4.append(k1[3])
        X5.append(k1[4])
        X6.append(k1[5])
    for l in range(0, len(labelPred)):
        if labelPred[l] == X[0]:
            gn1.append(l)
            ge1.append(('a', l))
            X1.append(F[l])
        elif labelPred[l] == X[1]:
            gn2.append(l)
            ge2.append(('b', l))
            X2.append(F[l])
        elif labelPred[l] == X[2]:
            gn3.append(l)
            ge3.append(('c', l))
            X3.append(F[l])
        elif labelPred[l] == X[3]:
            gn4.append(l)
            ge4.append(('d', l))
            X4.append(F[l])
        elif labelPred[l] == X[4]:
            gn5.append(l)
            ge5.append(('e', l))
            X5.append(F[l])
        elif labelPred[l] == X[5]:
            gn6.append(l)
            ge6.append(('f', l))
            X6.append(F[l])
    G1.add_nodes_from(gn1)
    G1.add_edges_from(ge1)
    G2.add_nodes_from(gn2)
    G2.add_edges_from(ge2)
    G3.add_nodes_from(gn3)
    G3.add_edges_from(ge3)
    G4.add_nodes_from(gn4)
    G4.add_edges_from(ge4)
    G5.add_nodes_from(gn5)
    G5.add_edges_from(ge5)
    G6.add_nodes_from(gn6)
    G6.add_edges_from(ge6)
    if os.path.exists(os.path.join(config.model_dir, 'GCN-1', "Target_Kgraph")) is not True:
        os.makedirs(os.path.join(config.model_dir, 'GCN-1', "Target_Kgraph"))
    nx.write_gpickle(G1, os.path.join(config.model_dir, 'GCN-1', "Target_Kgraph/graph_feature_1.gpickle"))
    nx.write_gpickle(G2, os.path.join(config.model_dir, 'GCN-1', "Target_Kgraph/graph_feature_2.gpickle"))
    nx.write_gpickle(G3, os.path.join(config.model_dir, 'GCN-1', "Target_Kgraph/graph_feature_3.gpickle"))
    nx.write_gpickle(G4, os.path.join(config.model_dir, 'GCN-1', "Target_Kgraph/graph_feature_4.gpickle"))
    nx.write_gpickle(G5, os.path.join(config.model_dir, 'GCN-1', "Target_Kgraph/graph_feature_5.gpickle"))
    nx.write_gpickle(G6, os.path.join(config.model_dir, 'GCN-1', "Target_Kgraph/graph_feature_6.gpickle"))
    np.save(os.path.join(config.model_dir, 'GCN-1', 'Target_Kgraph/X_1_fetures.npy'), X1)
    np.save(os.path.join(config.model_dir, 'GCN-1', 'Target_Kgraph/X_2_fetures.npy'), X2)
    np.save(os.path.join(config.model_dir, 'GCN-1', 'Target_Kgraph/X_3_fetures.npy'), X3)
    np.save(os.path.join(config.model_dir, 'GCN-1', 'Target_Kgraph/X_4_fetures.npy'), X4)
    np.save(os.path.join(config.model_dir, 'GCN-1', 'Target_Kgraph/X_5_fetures.npy'), X5)
    np.save(os.path.join(config.model_dir, 'GCN-1', 'Target_Kgraph/X_6_fetures.npy'), X6)

    if graph_path is not None:
        if os.path.exists(os.path.join(graph_path, 'Target_Kgraph')) is not True:
            os.makedirs(os.path.join(graph_path, 'Target_Kgraph'))
        nx.write_gpickle(G1, os.path.join(graph_path, "Target_Kgraph/graph_feature_1.gpickle"))
        nx.write_gpickle(G2, os.path.join(graph_path, "Target_Kgraph/graph_feature_2.gpickle"))
        nx.write_gpickle(G3, os.path.join(graph_path, "Target_Kgraph/graph_feature_3.gpickle"))
        nx.write_gpickle(G4, os.path.join(graph_path, "Target_Kgraph/graph_feature_4.gpickle"))
        nx.write_gpickle(G5, os.path.join(graph_path, "Target_Kgraph/graph_feature_5.gpickle"))
        nx.write_gpickle(G6, os.path.join(graph_path, "Target_Kgraph/graph_feature_6.gpickle"))
        np.save(os.path.join(graph_path, 'Target_Kgraph/X_1_fetures.npy'), X1)
        np.save(os.path.join(graph_path, 'Target_Kgraph/X_2_fetures.npy'), X2)
        np.save(os.path.join(graph_path, 'Target_Kgraph/X_3_fetures.npy'), X3)
        np.save(os.path.join(graph_path, 'Target_Kgraph/X_4_fetures.npy'), X4)
        np.save(os.path.join(graph_path, 'Target_Kgraph/X_5_fetures.npy'), X5)
        np.save(os.path.join(graph_path, 'Target_Kgraph/X_6_fetures.npy'), X6)


def preprocess(graph):
    A = nx.to_numpy_matrix(graph)
    I = np.eye(graph.number_of_nodes())
    A_hat = A + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = D_hat**0.5
    D_hat = np.matrix(np.diag(D_hat))
    D_hat = D_hat**-1
    return A_hat, D_hat















