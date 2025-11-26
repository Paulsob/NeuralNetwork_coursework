import math
import torch
from torch.autograd import Variable
from collections import namedtuple

class CPUForgetMult(torch.nn.Module):
    def __init__(self):
        super(CPUForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None):
        result = []
        forgets = f.split(1, dim=0)
        prev_h = hidden_init
        for i, h in enumerate((f * x).split(1, dim=0)):
            if prev_h is not None:
                h = h + (1 - forgets[i]) * prev_h
            h = h.view(h.size()[1:])
            result.append(h)
            prev_h = h
        return torch.stack(result)

class GPUForgetMult(torch.autograd.Function):
    configured_gpus = {}
    ptx = None

    def __init__(self):
        super(GPUForgetMult, self).__init__()

    def compile(self):
        self.forget_mult, self.bwd_forget_mult, self.stream = GPUForgetMult.configured_gpus[torch.cuda.current_device()]

    def forward(self, f, x, hidden_init=None):
        self.compile()
        seq_size, batch_size, hidden_size = f.size()
        result = f.new(seq_size + 1, batch_size, hidden_size)
        if hidden_init is not None:
            result[0, :, :] = hidden_init
        else:
            result = result.zero_()
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)
        self.forget_mult(grid=grid, block=(grid_hidden_size, 1),
                         args=[result.data_ptr(), f.data_ptr(), x.data_ptr(),
                               seq_size, batch_size, hidden_size],
                         stream=self.stream)
        self.save_for_backward(f, x, hidden_init)
        self.result = result
        return result[1:, :, :]

    def backward(self, grad_h):
        self.compile()
        f, x, hidden_init = self.saved_tensors
        h = self.result
        seq_size, batch_size, hidden_size = f.size()
        grad_f = f.new(*f.size())
        grad_x = f.new(*f.size())
        grad_h_init = f.new(batch_size, hidden_size)
        grid_hidden_size = min(hidden_size, 512)
        grid = (math.ceil(hidden_size / grid_hidden_size), batch_size)
        self.bwd_forget_mult(grid=grid, block=(grid_hidden_size, 1),
                             args=[h.data_ptr(), f.data_ptr(), x.data_ptr(),
                                   grad_h.data_ptr(), grad_f.data_ptr(),
                                   grad_x.data_ptr(), grad_h_init.data_ptr(),
                                   seq_size, batch_size, hidden_size],
                             stream=self.stream)
        if hidden_init is not None:
            return grad_f, grad_x, grad_h_init
        return grad_f, grad_x

class ForgetMult(torch.nn.Module):
    def __init__(self):
        super(ForgetMult, self).__init__()

    def forward(self, f, x, hidden_init=None, use_cuda=True):
        use_cuda = use_cuda and torch.cuda.is_available()
        if use_cuda:
            assert f.is_cuda and x.is_cuda
        if hidden_init is None:
            return GPUForgetMult()(f, x) if use_cuda else CPUForgetMult()(f, x)
        return GPUForgetMult()(f, x, hidden_init) if use_cuda else CPUForgetMult()(f, x, hidden_init)

if __name__ == '__main__':
    seq, batch, hidden = 3, 7, 19
    a = Variable(torch.rand(seq, batch, hidden).cuda(), requires_grad=True)
    forget = Variable(torch.rand(seq, batch, hidden).cuda(), requires_grad=True)
    last_h = Variable(torch.rand(batch, hidden).cuda(), requires_grad=True)

    resulta = ForgetMult()(forget, a, last_h, use_cuda=True)
    loss = resulta.pow(2).sum()
    loss.backward()

    x_grad_copy = a.grad.clone()

    a.grad.data *= 0
    forget.grad.data *= 0
    last_h.grad.data *= 0

    resultb = ForgetMult()(forget, a, last_h, use_cuda=False)
    loss = resultb.pow(2).sum()
    loss.backward()

    residual = (resulta - resultb)

    from torch.autograd import gradcheck
    inputs = [forget, a, last_h]
    test = gradcheck(ForgetMult(), inputs, eps=1e-4, atol=1e-2)
    print(test)
