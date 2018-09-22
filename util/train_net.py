from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import pdb
import os
import torch

def save_model(model, name, best_acc, epoch):
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'best_acc': best_acc,
        'start_epoch': epoch
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+name)

def save_model_quant(model, name, best_acc, epoch, all_G_kernels):
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'best_acc': best_acc,
        'start_epoch': epoch,
        'G_kernels': all_G_kernels
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+name)


def check_value(kernel):
    return kernel[1].data.abs().mean()

def optimization_step_float(model, loss, x_batch, y_batch, optimizer):
    """Make forward pass and update model parameters with gradients."""

    # forward pass
    x_batch, y_batch = Variable(x_batch.cuda()), Variable(y_batch.cuda(async=True))
    logits = model(x_batch)

    # compute logloss
    loss_value = loss(logits, y_batch)
    batch_loss = loss_value.data[0]

    # compute accuracies
    pred = F.softmax(logits)
    batch_accuracy = _accuracy(y_batch, pred, top_k=(1,))[0]

    # compute gradients
    optimizer.zero_grad()
    loss_value.backward()

    # update params
    optimizer.step()

    return batch_loss, batch_accuracy


def train(model, loss, optimization_step_fn,
          train_iterator, val_iterator, 
          params,
          n_epochs=30, steps_per_epoch=500, n_validation_batches=50,
          patience=10, threshold=0.01, lr_scheduler=None):
    """
    Train 'model' by minimizing 'loss' using 'optimization_step_fn'
    for parameter updates.
    """

    # collect losses and accuracies here
    all_losses = []
    best_acc, start_epoch, name = params

    is_reduce_on_plateau = isinstance(lr_scheduler, ReduceLROnPlateau)

    running_loss = 0.0
    running_accuracy = 0.0
    start_time = time.time()
    model.train()  # set train mode

    for epoch in range(0, n_epochs):

        # main training loop
        for step, (x_batch, y_batch) in enumerate(train_iterator, 1 + epoch*steps_per_epoch):

            batch_loss, batch_accuracy = optimization_step_fn(
                model, loss, x_batch, y_batch
            )

            running_loss += batch_loss
            running_accuracy += batch_accuracy


        # evaluation
        model.eval()
        test_loss, test_accuracy = _evaluate(
            model, loss, val_iterator, n_validation_batches
        )

        # collect evaluation information and print it
        all_losses += [(
            epoch,
            running_loss/steps_per_epoch, test_loss,
            running_accuracy/steps_per_epoch, test_accuracy
        )]

        # print('{0}  {1:.3f} {2:.3f}  {3:.3f} {4:.3f}  {5:.3f} {6:.3f}  {7:.3f}'.format(all_losses[-1], time.time() - start_time))
        print('{0}  {1:.3f}'.format(all_losses[-1], time.time() - start_time))


        if test_accuracy > best_acc:
            best_acc = test_accuracy
            print 'update best test accuracy as', best_acc
            save_model(model, name=name,
                       best_acc=best_acc, 
                       epoch=epoch + start_epoch)


        if lr_scheduler is not None:
            # possibly change the learning rate
            if not is_reduce_on_plateau:
                lr_scheduler.step()
            else:
                lr_scheduler.step(test_accuracy)

        running_loss = 0.0
        running_accuracy = 0.0
        start_time = time.time()
        model.train()

    return all_losses


def _accuracy(true, pred, top_k=(1,)):

    max_k = max(top_k)
    batch_size = true.size(0)

    _, pred = pred.topk(max_k, 1)
    pred = pred.t()
    correct = pred.eq(true.view(1, -1).expand_as(pred))

    result = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.div_(batch_size).data[0])

    return result

def _evaluate_quant(model, loss, val_iterator, n_validation_batches, kernels_list):
    W, G = kernels_list
    for i in range(len(W)):
        k_W = W[i]
        k_quant = G[i]    
        k_W.data, k_quant.data = k_quant.data, k_W.data

    test_loss, test_accuracy = _evaluate(
        model, loss, val_iterator, n_validation_batches
    )
        
    for i in range(len(W)):
        k_W = W[i]
        k_quant = G[i]    
        k_W.data, k_quant.data = k_quant.data, k_W.data

    return test_loss, test_accuracy

def _evaluate(model, loss, val_iterator, n_validation_batches):

    loss_value = 0.0
    accuracy = 0.0
    total_samples = 0

    for j, (x_batch, y_batch) in enumerate(val_iterator):

        x_batch = Variable(x_batch.cuda(), volatile=True)
        y_batch = Variable(y_batch.cuda(async=True), volatile=True)
        n_batch_samples = y_batch.size()[0]
        logits = model(x_batch)

        # compute logloss
        batch_loss = loss(logits, y_batch).data[0]

        # compute accuracies
        pred = F.softmax(logits)
        batch_accuracy = _accuracy(y_batch, pred, top_k=(1,))[0]

        loss_value += batch_loss*n_batch_samples
        accuracy += batch_accuracy*n_batch_samples
        total_samples += n_batch_samples

        if j >= n_validation_batches:
            break

    return loss_value/total_samples, accuracy/total_samples


def _is_early_stopping(all_losses, patience, threshold):
    """It decides if training must stop."""

    # get current and all past (validation) accuracies
    accuracies = [x[4] for x in all_losses]

    if len(all_losses) > (patience + 4):

        # running average with window size 5
        average = (accuracies[-(patience + 4)] +
                   accuracies[-(patience + 3)] +
                   accuracies[-(patience + 2)] +
                   accuracies[-(patience + 1)] +
                   accuracies[-patience])/5.0

        # compare current accuracy with
        # running average accuracy 'patience' epochs ago
        return accuracies[-1] < (average + threshold)
    else:
        # if not enough epochs to compare with
        return False
    

def train_eta(model, loss, optimization_step_fn,
          kernels_list,
          train_iterator, val_iterator, params, optimization_step_fn_eta,
          n_epochs=30, steps_per_epoch=500, n_validation_batches=50,
          patience=10, threshold=0.01, lr_scheduler=None,
          eta=1, eta_rate=1.05, m_epochs=80):
    """
    Train 'model' by minimizing 'loss' using 'optimization_step_fn'
    for parameter updates.
    """

    # collect losses and accuracies here
    all_losses = []

    is_reduce_on_plateau = isinstance(lr_scheduler, ReduceLROnPlateau)
    best_acc, start_epoch, name = params

    running_loss = 0.0
    running_accuracy = 0.0
    start_time = time.time()
    model.train()  # set train mode

    for epoch in range(0, n_epochs):

        if epoch < m_epochs:
            eta = eta_rate*eta
            

        # main training loop
        for step, (x_batch, y_batch) in enumerate(train_iterator, 1 + epoch*steps_per_epoch):
    
            if epoch < m_epochs:
                batch_loss, batch_accuracy = optimization_step_fn_eta(
                model, loss, x_batch, y_batch, eta)
            else:
                batch_loss, batch_accuracy = optimization_step_fn(
                model, loss, x_batch, y_batch)

            running_loss += batch_loss
            running_accuracy += batch_accuracy

        print 'check value of eta', eta

        # evaluation
        model.eval()
        test_loss, test_accuracy = _evaluate_quant(
            model, loss, val_iterator, n_validation_batches, kernels_list[:2])

        # collect evaluation information and print it
        all_losses += [(
            epoch + start_epoch,
            running_loss/steps_per_epoch, test_loss,
            running_accuracy/steps_per_epoch, test_accuracy
        )]

        # print('{0}  {1:.3f} {2:.3f}  {3:.3f} {4:.3f}  {5:.3f} {6:.3f}  {7:.3f}'.format(all_losses[-1], time.time() - start_time))
        print('{0}  {1:.3f}'.format(all_losses[-1], time.time() - start_time))
        
        if epoch == m_epochs:
            best_acc = 0

        # save model
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            print 'update best test accuracy as', best_acc
            save_model_quant(model, name=name,
                       best_acc=best_acc, 
                       epoch=epoch + start_epoch,
                       all_G_kernels=kernels_list[1])
        if epoch == n_epochs-1:
            print(best_acc)

        if lr_scheduler is not None:
            # possibly change the learning rate
            if not is_reduce_on_plateau:
                lr_scheduler.step()
            else:
                lr_scheduler.step(test_accuracy)

        running_loss = 0.0
        running_accuracy = 0.0
        start_time = time.time()
        model.train()

    return all_losses