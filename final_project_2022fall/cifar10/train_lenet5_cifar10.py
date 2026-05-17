# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.cifar10 import load_cifar10
from model.lenet5_conv3x3 import LeNet5
import matplotlib.pyplot as plt
from common.optimizer import *
from model.learning_function import *

# for reproducibility
np.random.seed(0)

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_cifar10(normalize=True, flatten=False, one_hot_label=True)
# print(x_train.shape, t_train.shape)

# a = x_train.T
# for i in range(50000):
#     for k in range(3):
#         new_train = a.T
# x_train = np.concatenate((x_train,new_train), axis=0)
# t_train = np.concatenate((t_train,t_train), axis=0)
# print(x_train.shape, t_train.shape)

# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network = LeNet5(input_dim=(3, 32, 32),
                  conv_param_1={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                  conv_param_2={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                  conv_param_3={'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 1},
                  weight_init_std=np.sqrt(2/(128*8*8)))

# optimizer = SGD()
optimizer = Adam(lr=0.001)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.001

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

best_test_acc = 0
path_dir = './ckpt'
# ----------------main--------------------------------
version = 'ver5'
project_name = version+"_"+"image_augmentation"

file_path = os.path.join("log", project_name + ".log")
final_param_file_name = version + "final_params.pkl"
best_param_file_name = version + "best_params.pkl"

if not os.path.isdir(path_dir):
    os.mkdir(path_dir)

# 딥러닝 학습
with open(file_path, "w", encoding="utf-8") as f:
    f.write("iter,train_acc,test_acc,loss\n")

    for i in range(iters_num):
        #학습률 하락
        learning_rate = learning_rate_ctrl(i,1000,learning_rate)
        learning_rate = learning_rate_ctrl(i,5000,learning_rate)
        optimizer.lr = learning_rate


        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # --- 데이터 증강 적용 ---
        if np.random.rand() > 0.5: # 50% 확률로 좌우 반전
            x_batch = flip_image(x_batch)
        x_batch = random_shift(x_batch, pad=2) # 매번 미세하게 이동
        
        # 기울기 계산
        grads = network.gradient(x_batch, t_batch) 
        params = network.params

        # 갱신
        optimizer.update(params, grads)
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 정해진 주기 마다 기록 저장
        if i % (iter_per_epoch/10) == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            
            # 최고 학습률 파라미터 저장
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                network.save_params(os.path.join(path_dir, best_param_file_name))
                print(f"새로운 최고 기록! {test_acc:.4f} 저장 완료.")
            
            print('iter: (%d:%d)' % (i, iters_num), '\ttrain acc: ', train_acc, '\ttest acc: ', test_acc)
            # save Terminal log
            f.write(f"{i},{train_acc:.5f},{test_acc:.5f},{loss:.5f}\n")
            f.flush()

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


# 마지막 파라미터 저장
network.save_params(os.path.join(path_dir, final_param_file_name))
print("Parameter Save Complete!")