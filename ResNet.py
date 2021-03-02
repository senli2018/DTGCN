import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F
from GCN_model import GraphConvolution
import numpy as np
import config
import graph_fun
import distance_fun

print("PyTorch Version: ",torch.__version__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = extractor(True)
        self.fc = nn.Linear(2048, 1024)
        self.gc1 = GraphConvolution(1024, 1024)
        self.gc2 = GraphConvolution(1024, 1024)

    def forward(self, x, source_length=0, source_labels=None, source_centers=None):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        features = x.cpu().data.numpy()
        if source_centers is None:
            source_centers = get_centers(features[:source_length], source_labels)
        target_labels = []
        for feature in features[source_length:]:
            dis_list = []
            for center in source_centers:
                dis_list.append(distance_fun.euclidean_distance(feature, center))
            target_labels.append(np.argmin(dis_list))
        if source_labels is not None:
            adj = graph_fun.domain_cluster_graph(source_labels, target_labels, init_graph=None)
        else:
            adj = graph_fun.supervised_graph(target_labels)

        if source_length > 0:
            adj[:source_length, :source_length] = np.eye(source_length)
        A, D = graph_fun.process_graph(adj)
        A, D = torch.tensor(A, dtype=torch.float32, requires_grad=True).cuda(), torch.tensor(D, dtype=torch.float32).cuda()
        x = self.gc1(x, A, D)
        x = self.gc2(x, A, D)
        return x, A, features


def get_centers(features, labels):
    centers = np.zeros([max(labels) + 1, features.shape[1]])
    for i in range(len(labels)):
        centers[labels[i]] += features[i]
    return centers



def extractor(pre_training=False):
    model = resnet18(pretrained=pre_training)
    return model


if __name__=='__main__':
    #model = torchvision.models.resnet50()
    model = Net()
    # print(model)

    input = torch.randn(20, 3, 224, 224)
    out = model(input)
    print(out.shape)
