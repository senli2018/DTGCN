import os

import config
import train_fun
import utils_fun

mode = 2

if __name__ == '__main__':
    if os.path.exists(config.model_dir) is not True:
        os.makedirs(config.model_dir)
    if os.path.exists('label_true.npy') is not True:
        utils_fun.get_true_label()

    if mode == 1:
        the_mode = 1
        if os.path.exists(os.path.join(config.model_dir, config.model_name + '-' + str(the_mode))) is not True:
            os.makedirs(os.path.join(config.model_dir, config.model_name + '-' + str(the_mode)))
        print('Start to Extract Features')
        train_fun.train1(0.45, the_mode)
        print("Finish Extracting Features")
    if mode == 2:
        if os.path.exists(os.path.join(config.model_dir, 'GCN-1')) is not True:
            os.makedirs(os.path.join(config.model_dir, 'GCN-1'))
        print('Start to train GCN')
        train_fun.train2()
        print('Finish Training GCN')























