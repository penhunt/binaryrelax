import torch.nn as nn
import torch.optim as optim

def get_model(model, learning_rate=1e-3, weight_decay=1e-4):

    # set the first layer not trainable
    # model.features.conv0.weight.requires_grad = False

    # all fc layers
    weights = [
        p for n, p in model.named_parameters()
        if 'classifier.weight' in n
    ]
    biases = [model.classifier.bias] 
    
    # all conv layers except the first
    weights_to_be_quantized = [
        p for n, p in model.named_parameters()
        # if 'conv' in n and 'conv0' not in n
        if 'conv' in n and 'weight' in n
    ]

    biases_conv = [
        p for n, p in model.named_parameters()
        # if 'conv' in n and 'conv0' not in n
        if 'conv' in n and 'bias' in n
    ]    
    biases = biases + biases_conv

    # parameters of batch_norm layers
    bn_weights = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'weight' in n
    ]
    bn_biases = [
        p for n, p in model.named_parameters()
        if 'norm' in n and 'bias' in n
    ]

    params = [
        {'params': weights, 'weight_decay': weight_decay},
        {'params': weights_to_be_quantized},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]
    optimizer = optim.Adam(params, lr=learning_rate)

    loss = nn.CrossEntropyLoss().cuda()
    model = model.cuda()  # move the model to gpu
    return model, loss, optimizer

def get_model2(model, learning_rate=1e-3, weight_decay=1e-4):

    # set the first layer not trainable
    # model.features.conv0.weight.requires_grad = False

    # all fc layers
    weights = [
        p for n, p in model.named_parameters()
        if 'weight' in n and 'conv' not in n
    ]

    # all conv layers
    weights_to_be_quantized = [
        p for n, p in model.named_parameters()
        # if 'conv' in n and 'conv0' not in n
        if 'conv' in n and 'weight' in n
    ]

    biases = [
        p for n, p in model.named_parameters()
        if 'bias' in n
    ]    

    params = [
        {'params': weights, 'weight_decay': weight_decay},
        {'params': weights_to_be_quantized, 'weight_decay': weight_decay},
        {'params': biases,  'weight_decay': weight_decay}
    ]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9)

    loss = nn.CrossEntropyLoss().cuda()
    model = model.cuda()  # move the model to gpu
    return model, loss, optimizer