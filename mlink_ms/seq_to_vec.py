from mindspore import nn, ops


class Seq2VecMLink(nn.Cell):
    
    def __init__(self, input_len, output_len, 
                 hidden=2, output_activation="softmax"):
        super(Seq2VecMLink, self).__init__()

        self.emb = nn.Embedding(input_len, hidden*output_len)
        self.lstm = nn.LSTM(hidden*output_len, hidden*output_len)
        self.flatten = nn.Flatten()
        self.dense = nn.Dense((output_len*hidden)**2, output_len)
        if output_activation == "softmax":
            self.out_act = nn.Softmax()
        if output_activation == "sigmoid":
            self.out_act = nn.Sigmoid()

    def construct(self, x):
        x = self.emb(x)
        x, hx = self.lstm(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.out_act(x)
        return x
    

if __name__ == "__main__":
    import mindspore as ms
    import numpy as np

    net = Seq2VecMLink(20, 10)
    x = ms.Tensor(np.ones([1, 20]), ms.int16)
    y = net(x)
    print(y.shape)