import numpy as np

def learning_rate_ctrl(i,branch=1000, learning_rate=0.001):
    if(i == branch):
        learning_rate *= 0.1
        msg = f"\n--- Learning Rate decayed to {learning_rate} ---\n"
        print(msg)
    return learning_rate

def random_shift(x, pad=2):
    # 이미지를 패딩 후 랜덤하게 자름
    n, c, h, w = x.shape
    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    
    new_x = np.zeros_like(x)
    for i in range(n):
        top = np.random.randint(0, 2*pad + 1)
        left = np.random.randint(0, 2*pad + 1)
        new_x[i] = x_padded[i, :, top:top+h, left:left+w]
    return new_x

def flip_image(x):
    # x shape: (Batch, Channel, H, W)
    return x[:, :, :, ::-1]
