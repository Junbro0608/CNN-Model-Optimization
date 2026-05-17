# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class LeNet5:
    def __init__(self, input_dim=(1, 32, 32),
                 conv_param_1={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_2={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_3={'filter_num': 128, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 weight_init_std=0.01):
        

        # 가중치 초기화 filter_num, input_dim[0], filter_size, filter_size
        self.params = {}
        # Conv1
        self.params['W1_1'] = weight_init_std * np.random.randn(32, 3, 3, 3)
        self.params['b1_1'] = np.zeros(32)
        self.params['W1_2'] = weight_init_std * np.random.randn(32, 32, 3, 3)
        self.params['b1_2'] = np.zeros(32)
        self.params['gamma1'] = np.ones(32768) # BatchNorm용
        self.params['beta1'] = np.zeros(32768) # BatchNorm용
        
        # Conv3
        self.params['W3_1'] = weight_init_std * np.random.randn(64, 32, 3, 3)
        self.params['b3_1'] = np.zeros(64)
        self.params['W3_2'] = weight_init_std * np.random.randn(64, 64, 3, 3)
        self.params['b3_2'] = np.zeros(64)
        self.params['gamma2'] = np.ones(16384) # BatchNorm용 (32->28->(pool)->14->Conv3(5x5)->10 이므로 64*10*10)
        self.params['beta2'] = np.zeros(16384) # BatchNorm용
        # Conv5
        self.params['W5_1'] = weight_init_std * np.random.randn(128, 64, 3, 3)
        self.params['b5_1'] = np.zeros(128)
        self.params['W5_2'] = weight_init_std * np.random.randn(128, 128, 3, 3)
        self.params['b5_2'] = np.zeros(128)

        # Affine 계층
        affine_snum = 128 * 8 * 8
        self.params['W6'] = np.sqrt(2/affine_snum) * np.random.randn(affine_snum, 480)
        self.params['b6'] = np.zeros(480)
        self.params['W7'] = np.sqrt(2/480) * np.random.randn(480, 480)
        self.params['b7'] = np.zeros(480)
        self.params['W8'] = np.sqrt(2/480) * np.random.randn(480, 10)
        self.params['b8'] = np.zeros(10)


        # 계층 생성
        self.layers = OrderedDict()
        # C1 : 컨볼루션 연산
        self.layers['Conv1_1'] = Convolution(self.params['W1_1'], self.params['b1_1'], conv_param_1['stride'], conv_param_1['pad'])
        self.layers['Conv1_2'] = Convolution(self.params['W1_2'], self.params['b1_2'], conv_param_1['stride'], conv_param_1['pad'])
        self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['ELU_1'] = Elu()
        # S2 : 풀링 계층 (LeNet에서는 평균풀링을 사용했으나, 여기서는 최대풀링)
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # C3 : 컨볼루션 연산
        self.layers['Conv3_1'] = Convolution(self.params['W3_1'], self.params['b3_1'],conv_param_2['stride'], conv_param_2['pad'])
        self.layers['Conv3_2'] = Convolution(self.params['W3_2'], self.params['b3_2'],conv_param_2['stride'], conv_param_2['pad'])
        self.layers['BatchNorm2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['ELU_2'] = Elu()
        # S4 : 풀링 계층
        self.layers['Pool4'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # C5 : 컨볼루션 연산
        self.layers['Conv5_1'] = Convolution(self.params['W5_1'], self.params['b5_1'],conv_param_3['stride'], conv_param_3['pad'])
        self.layers['Conv5_2'] = Convolution(self.params['W5_2'], self.params['b5_2'],conv_param_3['stride'], conv_param_3['pad'])
        self.layers['ELU_3'] = Elu()
        # F6 : Affine 계층 120 -> 84
        self.layers['Affine6'] = Affine(self.params['W6'], self.params['b6'])
        self.layers['ELU_4'] = Elu()
        self.layers['Dropout1'] = Dropout(0.5)
        # F7
        self.layers['Affine7'] = Affine(self.params['W7'], self.params['b7'])
        self.layers['ELU_5'] = Elu()
        self.layers['Dropout2'] = Dropout(0.5)
        # F8 : Affine 계층 + 활성화 함수는 softmax  84 -> 10
        self.layers['Affine8'] = Affine(self.params['W8'], self.params['b8'])


        self.last_layer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        #C1
        grads['W1_1'], grads['b1_1'] = self.layers['Conv1_1'].dW, self.layers['Conv1_1'].db
        grads['W1_2'], grads['b1_2'] = self.layers['Conv1_2'].dW, self.layers['Conv1_2'].db
        grads['gamma1'], grads['beta1'] = self.layers['BatchNorm1'].dgamma, self.layers['BatchNorm1'].dbeta
        #C3
        grads['W3_1'], grads['b3_1'] = self.layers['Conv3_1'].dW, self.layers['Conv3_1'].db
        grads['W3_2'], grads['b3_2'] = self.layers['Conv3_2'].dW, self.layers['Conv3_2'].db
        grads['gamma2'], grads['beta2'] = self.layers['BatchNorm2'].dgamma, self.layers['BatchNorm2'].dbeta
        #C5
        grads['W5_1'], grads['b5_1'] = self.layers['Conv5_1'].dW, self.layers['Conv5_1'].db
        grads['W5_2'], grads['b5_2'] = self.layers['Conv5_2'].dW, self.layers['Conv5_2'].db

        #Affine
        grads['W6'], grads['b6'] = self.layers['Affine6'].dW, self.layers['Affine6'].db
        grads['W7'], grads['b7'] = self.layers['Affine7'].dW, self.layers['Affine7'].db
        grads['W8'], grads['b8'] = self.layers['Affine8'].dW, self.layers['Affine8'].db

        return grads

    def save_params(self, file_name="params_Lenet.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params_Lenet.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

            j = 1

        for i, key in enumerate(['Conv1', 'Conv3', 'Conv5', 'Affine6', 'Affine7', 'Affine8']):
            self.layers[key].W = self.params['W' + str(i + j)]
            self.layers[key].b = self.params['b' + str(i + j)]
            if j < 3:
                j += 1