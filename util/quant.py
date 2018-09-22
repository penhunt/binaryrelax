import torch
from torch.autograd import Variable
import torch.nn.functional as F
from train_net import _accuracy
import pdb

def check_value(kernel):
    print(kernel[1].data.abs().max())

def quantize_bw(kernel):
    """
    binary quantization
    Return quantized weights of a layer.
    """
    delta = kernel.abs().mean()
    sign = kernel.sign().float()
    return sign*delta

def quantize_tnn(kernel):
    """
    ternary quantization
    Return quantized weights of a layer.
    """
    data = kernel.abs()
    delta = 0.7*data.mean()
    delta = min(delta, 100.0)
    index = data.ge(delta).float()
    sign = kernel.sign().float()
    scale = (data*index).mean()
    return scale*index*sign
   

def optimization_step_eta(model, loss, x_batch, y_batch, optimizer_list, eta):
    """
    steps:
        1. quantized W to get G
        2. use G in forward
        3. update W with W - grad(G)

    optimizer_list:  
        optimizer: optimizer tool for NN
        optimizer_qunat: lr=0, used to recoder G
    """

    optimizer, optimizer_quant = optimizer_list
    x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))

    # get all kernels
    all_W_kernels = optimizer.param_groups[1]['params']
    all_G_kernels = optimizer_quant.param_groups[0]['params']

    for i in range(len(all_W_kernels)):
        k_W = all_W_kernels[i]
        k_G = all_G_kernels[i]
        V = k_W.data
        # ternary 
        # k_G.data = quantize_tnn(V)
        # binary
        k_G.data = (eta*quantize_bw(V)+V)/(1+eta)
        k_W.data, k_G.data = k_G.data, k_W.data

    # forward pass using quantized model
    logits = model(x_batch)
    # compute logloss
    loss_value = loss(logits, y_batch)
    batch_loss = loss_value.data[0]
    # compute accuracies
    pred = F.softmax(logits)
    batch_accuracy = _accuracy(y_batch, pred, top_k=(1,))[0]
    optimizer.zero_grad()
    # compute grads
    loss_value.backward()

    for i in range(len(all_W_kernels)):
        k_W = all_W_kernels[i]
        k_G = all_G_kernels[i]
        k_W.data, k_G.data = k_G.data, k_W.data

    # update parameters
    optimizer.step()

    return batch_loss, batch_accuracy


def optimization_step(model, loss, x_batch, y_batch, optimizer_list):
    optimizer, optimizer_quant = optimizer_list
    x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))

    all_W_kernels = optimizer.param_groups[1]['params']
    all_G_kernels = optimizer_quant.param_groups[0]['params']

    for i in range(len(all_W_kernels)):
        k_W = all_W_kernels[i]
        k_G = all_G_kernels[i]
        V = k_W.data
        # ternary 
        # k_G.data = quantize_tnn(V)
        # binary
        k_G.data = quantize_bw(V)
        k_W.data, k_G.data = k_G.data, k_W.data

    logits = model(x_batch)
    loss_value = loss(logits, y_batch)
    batch_loss = loss_value.data[0]
    pred = F.softmax(logits)
    batch_accuracy = _accuracy(y_batch, pred, top_k=(1,))[0]
    optimizer.zero_grad()
    loss_value.backward()

    for i in range(len(all_W_kernels)):
        k_W = all_W_kernels[i]
        k_G = all_G_kernels[i]
        k_W.data, k_G.data = k_G.data, k_W.data

    optimizer.step()

    return batch_loss, batch_accuracy