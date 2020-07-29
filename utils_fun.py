from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import config
import numpy as np
import joblib

Target_dataset = dataset.ImageFolder(config.target_dataset_path, transform=config.testTransform)
target_dataloader = DataLoader(Target_dataset, 1, shuffle=False)


def get_true_label():
    """
    获取真实label的文件，方便后续调用
    """
    labels_true = []
    for data in target_dataloader:
        inputs, label = data
        print(label)
        labels_true.append(label[0].item())
    np.save('label_true.npy', labels_true)
    print(len(labels_true))
    print(labels_true)


def test_labels(l):
    l_raw = np.load('label_true.npy')
    class1 = config.class1_num
    class2 = config.class2_num + class1
    class3 = config.class3_num + class2
    class4 = config.class4_num + class3
    class5 = config.class5_num + class4
    class6 = config.class6_num + class5
    for index, i in enumerate(l):
        if i <= class1:
            l[index] = l_raw[0]
            # print(l_raw[0])
        elif class1 < i <= class2:
            l[index] = l_raw[class1 + 1]
            # print(l)
        elif class2 < i <= class3:
            l[index] = l_raw[class2 + 1]
        elif class3 < i <= class4:
            l[index] = l_raw[class3 + 1]
        elif class4 < i <= class5:
            l[index] = l_raw[class4 + 1]
        elif class5 < i <= class6:
            l[index] = l_raw[class5 + 1]
    return l


def test_acc(labels_):
    labels_true = np.load('label_true.npy')
    label_pred = labels_

    i1 = []
    i2 = []
    i3 = []
    i4 = []
    i5 = []
    i6 = []
    for i, label in enumerate(label_pred):
        if label == 0:
            i1.append(i)
        elif label == 1:
            i2.append(i)
        elif label == 2:
            i3.append(i)
        elif label == 3:
            i4.append(i)
        elif label == 4:
            i5.append(i)
        elif label == 5:
            i6.append(i)
    i1 = test_labels(i1)
    i2 = test_labels(i2)
    i3 = test_labels(i3)
    i4 = test_labels(i4)
    i5 = test_labels(i5)
    i6 = test_labels(i6)
    label1 = max(i1, key=i1.count)
    label2 = max(i2, key=i2.count)
    label3 = max(i3, key=i3.count)
    label4 = max(i4, key=i4.count)
    label5 = max(i5, key=i5.count)
    label6 = max(i6, key=i6.count)
    labels_pred = []
    for i in label_pred:
        if i == 0:
            labels_pred.append(label1)
        if i == 1:
            labels_pred.append(label2)
        if i == 2:
            labels_pred.append(label3)
        if i == 3:
            labels_pred.append(label4)
        if i == 4:
            labels_pred.append(label5)
        if i == 5:
            labels_pred.append(label6)
    acc = ((labels_pred == labels_true).sum())/len(labels_pred)
    return acc, labels_pred


def save_txt_files(path, the_list):
    f = open(path, 'a')
    for i in the_list:
        f.write(str(i) + '\n')
    f.close()


def save_txt_file(path, the_str, mode):
    f = open(path, mode)
    f.write(the_str + '\n')
    f.close()


if __name__ == '__main__':
    get_true_label()





