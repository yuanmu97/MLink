from mindspore import nn, ops


class Vec2VecMLink(nn.Cell):
    
    def __init__(self, input_len, output_len, 
                 hidden=2, dropout=0.5, output_activation="softmax"):
        super(Vec2VecMLink, self).__init__()
        self.dense1 = nn.Dense(input_len, output_len*hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.dense2 = nn.Dense(output_len*hidden, output_len)
        if output_activation == "softmax":
            self.out_act = nn.Softmax()
        if output_activation == "sigmoid":
            self.out_act = nn.Sigmoid()

    def construct(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.out_act(x)
        return x
    

if __name__ == "__main__":
    import mindspore as ms
    import numpy as np

    net = Vec2VecMLink(20, 10)
    x = ms.Tensor(np.ones([1, 20]), ms.float32)
    y = net(x)
    print(y.shape)