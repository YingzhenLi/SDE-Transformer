"""
Bayesian DL via SDEs

#SDEs as building blocks for Bayesian neural networks

Training util functions
"""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.calibration import calibration_curve, expected_calibration_error as ece

EPS = 1e-5  # define a small constant for numerical stability control
BAYES_BLOCK_TYPE = ['mfvi', 'sgpa', 'sde', 'sparsecorrsde', 'mhsde']

def to_numpy(x):
    return x.detach().cpu().numpy() # convert a torch tensor to a numpy array

def net_hasattr(net, attr_name):
    # check if any layer in net has attribute attr_name
    for m in net.modules():
        if hasattr(m, attr_name):
            return True
    return False

def train_step(net, opt, dataloader, device, beta=1.0, compute_kl=False, compute_map=False):
    logs = []
    N_data = len(dataloader.dataset)
    for _, (x, y) in enumerate(dataloader):
        x = x.to(device); y = y.to(device)
        opt.zero_grad() # opt is the optimiser
        y_logit = net(x)
        # training accruacy (on a mini-batch)
        pred = y_logit.data.max(1, keepdim=True)[1] # get the index of the max logit
        acc = pred.eq(y.data.view_as(pred)).float().cpu().mean() * 100  # accuracy in percentage
        # training loss
        nll = F.nll_loss(F.log_softmax(y_logit, dim=-1), y)
        if compute_kl:
            kl = sum(m.kl_divergence() for m in net.modules() if hasattr(m, "kl_divergence"))
            loss = nll + beta * kl / N_data
            logs.append([to_numpy(nll), to_numpy(acc), to_numpy(kl)])
        elif compute_map:
            log_prior = sum(m.log_prior() for m in net.modules() if hasattr(m, "log_prior"))
            loss = nll - beta * log_prior / N_data
            logs.append([to_numpy(nll), to_numpy(acc), to_numpy(-log_prior)])
        else:
            loss = nll
            logs.append([to_numpy(nll), to_numpy(acc)])
        loss.backward()
        opt.step()
    return np.array(logs).mean(0)

# define the training function
def train_net(net, opt, dataloader, device, N_epochs=2000, beta=1.0, compute_kl=False, 
              compute_map=False, verbose=True):
    if N_epochs <= 0:
        return 0
    net.train()
    logs = []
    if compute_kl is not False:
        compute_kl = net_hasattr(net, 'kl_divergence')
    assert compute_kl in [True, False]
    print('compute_kl = {}'.format(compute_kl))
    if compute_kl:
        compute_map = False
    if compute_map is not False:
        compute_map = net_hasattr(net, 'log_prior')
    assert compute_map in [True, False]
    print('compute_map = {}'.format(compute_map))
    assert not (compute_kl and compute_map)

    logs_name = ['nll', 'acc']
    if compute_kl:
        logs_name.append('kl')
    if compute_map:
        logs_name.append('neg_log_prior')
    tepoch = tqdm(range(N_epochs), unit=' epoch')
    for i in tepoch:
        tepoch.set_description("Epoch {}".format(i+1))
        logs_epoch = train_step(net, opt, dataloader, device, beta, compute_kl, compute_map)
        if verbose:
            loss = {}
            for j in range(len(logs_name)):
                loss[logs_name[j]] = logs_epoch[j]
            tepoch.set_postfix(loss)
        logs.append(logs_epoch)
    return np.concatenate(logs, axis=0)

def predict(model, x, K):
    if K == 1:
        return F.softmax(model(x), dim=-1)
    # MC sampling with K>1 samples
    pred = []
    for _ in range(K):
        pred.append(F.softmax(model(x), dim=-1).unsqueeze(-1))
    torch.cuda.empty_cache()
    return torch.cat(pred, dim=-1).mean(-1)

def evaluate(model, dataloader, device, block_type, K=10):
    accuracy = 0
    probs_pred = []
    y_true = []
    if block_type not in BAYES_BLOCK_TYPE:
        K = 1   # deterministic prediction
    if K == 1:
        model.eval()
    for x, y in dataloader:
        x = x.to(device); y = y.to(device)
        probs_pred.append(to_numpy(predict(model, x, K)))
        y_true.append(to_numpy(y))
    probs_pred = np.concatenate(probs_pred); y_true = np.concatenate(y_true)
    accuracy = np.equal(np.argmax(probs_pred, -1), y_true).astype(float).mean() * 101
    p, f, w = calibration_curve(probs_pred, y_true, num_bins=10)
    ece_error = ece(p, f, w) * 100  # ECE in percentage
    return accuracy, ece_error

