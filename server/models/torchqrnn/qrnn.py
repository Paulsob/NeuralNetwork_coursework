import torch
from torch import nn
from torch.autograd import Variable
#ф-ция обновления скрытых состояний
if __name__ == '__main__':
    from forget_mult import ForgetMult
else:
    from .forget_mult import ForgetMult


class QRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=1, output_gate=True, use_cuda=True):
        super(QRNNLayer, self).__init__()

        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate
        self.use_cuda = use_cuda

        self.linear = nn.Linear(self.window * self.input_size, 3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)

    def reset(self):
        self.prevX = None

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()

        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            Xm1 = []
            Xm1.append(self.prevX if self.prevX is not None else X[:1, :, :] * 0)
            if len(X) > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            source = torch.cat([X, Xm1], 2)

        Y = self.linear(source)
        if self.output_gate:
            Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            Y = Y.view(seq_len, batch_size, 2 * self.hidden_size)
            Z, F = Y.chunk(2, dim=2)

        Z = torch.nn.functional.tanh(Z)
        F = torch.nn.functional.sigmoid(F)

        if self.zoneout:
            if self.training:
                mask = Variable(F.data.new(*F.size()).bernoulli_(1 - self.zoneout), requires_grad=False)
                F = F * mask
            else:
                F *= 1 - self.zoneout

        Z = Z.contiguous()
        F = F.contiguous()

        C = ForgetMult()(F, Z, hidden, use_cuda=self.use_cuda)

        if self.output_gate:
            H = torch.nn.functional.sigmoid(O) * C
        else:
            H = C

        if self.window > 1 and self.save_prev_x:
            self.prevX = Variable(X[-1:, :, :].data, requires_grad=False)

        return H, C[-1:, :, :]

class QRNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, layers=None, **kwargs):
        assert bidirectional == False, 'Bidirectional QRNN is not yet supported'
        assert batch_first == False, 'Batch first mode is not yet supported'
        assert bias == True, 'Removing underlying bias is not yet supported'

        super(QRNN, self).__init__()

        self.layers = torch.nn.ModuleList(layers if layers else [QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers) if layers else num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def reset(self):
        [layer.reset() for layer in self.layers]

    def forward(self, input, hidden=None):
        next_hidden = []

        for i, layer in enumerate(self.layers):
            input, hn = layer(input, None if hidden is None else hidden[i])
            next_hidden.append(hn)

            if self.dropout != 0 and i < len(self.layers) - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])

        return input, next_hidden


if __name__ == '__main__':
    seq_len, batch_size, hidden_size, input_size = 7, 20, 256, 32
    size = (seq_len, batch_size, input_size)
    X = torch.autograd.Variable(torch.rand(size), requires_grad=True).cuda()
    qrnn = QRNN(input_size, hidden_size, num_layers=2, dropout=0.4)
    qrnn.cuda()
    output, hidden = qrnn(X)
    assert list(output.size()) == [7, 20, 256]
    assert list(hidden.size()) == [2, 20, 256]

    seq_len, batch_size, hidden_size = 2, 2, 16
    seq_len, batch_size, hidden_size = 35, 8, 32
    size = (seq_len, batch_size, hidden_size)
    X = Variable(torch.rand(size), requires_grad=True).cuda()
    print(X.size())

    qrnn = QRNNLayer(hidden_size, hidden_size)
    qrnn.cuda()
    Y, _ = qrnn(X)

    qrnn.use_cuda = False
    Z, _ = qrnn(X)

    diff = (Y - Z).sum().data[0]
    print('Total difference between QRNN(use_cuda=True) and QRNN(use_cuda=False) results:', diff)
    assert diff < 1e-5, 'CUDA and non-CUDA QRNN layers return different results'

    from torch.autograd import gradcheck
    inputs = [X,]
    test = gradcheck(QRNNLayer(hidden_size, hidden_size).cuda(), inputs)
    print(test)
