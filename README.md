# DTGCN

This page shows the original code and data samples of the paper 'Multi-Stage Malaria Parasites Recognition by Deep Transfer Graph Convolutional Network'.


The image samples can be downloaded in [Dataset](https://data.mendeley.com/datasets/xvs55d4rcz/draft?a=6223e44e-04b8-4705-91d9-bf98665c6194).

The code in this page is for multi-stage malaria prasites recognition. Please revised the parameters in 'config.py' when conduct on other datasets.

A CNN feature collection and trained model are in '/Model_and_Log/'.

#Detailed Setting
We have released the code under a OSI compliant license (MIT) with a license file in GitHub (https://github.com/senli2018/DTGCN) and mentioned in our paper.

The code and trained model can be downloaded from GitHub, and the detail information is described below.

Running Environment

	Operating System: Windows 10;

	GPU: Nvidia Geforce 2080Ti GPU;

	Deep learning framework: PyTorch 1.0.0 in Python 3.6.0;

Requirements:

	Pytorch 1.0.0;

	Torchvision 0.4.1;

	Scipy 1.1.0;

	Numpy 1.17.4;

Parameter in code has been described in the GitHub, and can be directly run. The detail parameters are introduced as follow:

For the CNN feature learning, ResNet18 architecture is firstly optimized in 50 epochs, and then the parameters are fixed in the following training. In this section, the batch size is set as 15 for source and target data with learning rate of 2e-5 which multiply 0.1 in every 50 iterations, the margin parameter m=5 in Eq. (3) and the balance parameter, \lambda=0.45 for L_cnn and L_mmd to formulate the overall loss function L=L_cnn+\lambda* L_mmd of CNN.

For the training details in GCN, the two graph convolution layers are trained with the output CNN features of the pre-trained CNN. 

For both of CNN and GCN training, the DTGCN is implemented by the PyTorch framework with GTX2080Ti GPU and employ Adam optimizer to optimize the parameters of the network. As for GCN, the learning rate is 2e-7 and is optimized by manually gradient descent operation. And the settings of large-scale malaria parasites recognition are also following this setting.

Finally, we introduce how to run DTGCN code.
1.	Download the DTGCN code from GitHub (https://github.com/senli2018/DTGCN).

2.	Download the dataset from Mendeley and release it into the root dir of code.

3.	The data consist three tasks of ‘1_multistage_malaria_classification’, ‘2_Unseen_Malaria_classification’ and ‘3_babesia_classification’, it can change the path variables in ‘config.py’, such as 

(a)	Multi-stage malaria parasites recognition task:

source_dataset_path = '1_multistage_malaria_classification/train'

target_dataset_path = '1_multistage_malaria_classification/test'	

(b)	Large scale malaria classification task:

source_dataset_path = '2_Unseen_Malaria_classification/train'

target_dataset_path = '2_Unseen_Malaria_classification/test'	

(c)	babesia_classification

source_dataset_path = '3_babesia_classification '

target_dataset_path = '3_babesia_classification '

4.Taking multi-stage malaria parasites recognition task as an example, the CNN feature learning module is time-consuming in training, thus we provide a pre-trained CNN features and source centre files in folder of ‘Model_and_Log’, which supports directly training GCN module for the multi-stage malaria parasites recognition task. Just run ‘main.py’, and the programme will automatically load existing models and start train the GCN, with outputting loss and predicted results along with epochs. It will be soon convergence in 10 epochs, and reported results can be obtained.
