import os
import torchvision.transforms as transforms

model_path = r'Model_and_Log/model.pkl'

class_num = 6

# resnet
source_batch_size = 30
target_batch_size = 10


source_path = r'1_multistage_malaria_classification/train'
target_path = r'1_multistage_malaria_classification/test'

epoches = 500
k = 10

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])











