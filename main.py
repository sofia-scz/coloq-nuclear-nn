import torch
from torch.nn import Sequential, Linear, Softplus, Tanh, MSELoss
from torch import tensor
from torch import float as tfloat
from torch.optim import RAdam
import numpy as np
import pandas as pd
from time import time
import os
from process_database import df
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.font_manager as fm
import shap

# set up gfx 1010
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
device = 'cuda'

# setup plt
fm = fm.fontManager.addfont(path='/home/sofia/fonts/Ubuntu-Regular.ttf')
plt.rc('font', family='Ubuntu', size='14')
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 200
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=2)
red, green, yellow = '#c33', '#395', '#dc2'

# set up random seed
np.random.seed(2245)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# set up model
model_in = 8
model_out = 1

h_nodes, bn_nodes, e_layers, n_layers = 50, 10, 5, 20
layers = [Linear(model_in, h_nodes)] \
    + [Linear(h_nodes, h_nodes), Softplus()]*n_layers \
    + [Linear(h_nodes, model_out)]


ann = Sequential(*layers).to(device)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load data
ds = df
datalen = len(ds)

# convert to tensors
ds_tensor = tensor(ds.values, dtype=tfloat, device=device)
data_x = ds_tensor[:, :-2]
data_f = ds_tensor[:, -2]
sigma = ds_tensor[:, -1] ** .2

# interpolation training set
nbase = 50
ntrain = 500
nval = datalen - ntrain
choice = np.random.choice(range(nbase, datalen), size=ntrain-nbase, replace=False)
indexes = np.concatenate((np.arange(nbase), choice))
train_indexes = sorted(indexes)
val_indexes = [i for i in range(datalen) if not i in train_indexes]
train_f = data_f[train_indexes]
train_s = sigma[train_indexes]
val_f = data_f[val_indexes]
val_s = sigma[val_indexes]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# training

# def error criteria
mse = MSELoss()

# constants
zeros = torch.zeros(datalen, dtype=tfloat, device=device)
izeros = torch.zeros(ntrain, dtype=tfloat, device=device)
vzeros = torch.zeros(nval, dtype=tfloat, device=device)


# def GK penalty matrix
GK_mat = torch.zeros((datalen, datalen), dtype=tfloat, device=device)
for i in range(datalen):
    try:
        z, n = ds.iloc[i]['Z'], ds.iloc[i]['N']
        if z <= n:
            p1 = ds.loc[(ds['Z'] == z-2) & (ds['N'] == n+2)].index[0]
            p2 = ds.loc[(ds['Z'] == z-1) & (ds['N'] == n)].index[0]
            p3 = ds.loc[(ds['Z'] == z) & (ds['N'] == n+1)].index[0]
            n1 = ds.loc[(ds['Z'] == z-2) & (ds['N'] == n+1)].index[0]
            n2 = ds.loc[(ds['Z'] == z-1) & (ds['N'] == n+2)].index[0]
            GK_mat[i, p1], GK_mat[i, p2], GK_mat[i, p3] = (1,)*3
            GK_mat[i, i], GK_mat[i, n1], GK_mat[i, n2] = (-1,)*3
        elif z > n:
            p1 = ds.loc[(ds['Z'] == z+2) & (ds['N'] == n-2)].index[0]
            p2 = ds.loc[(ds['Z'] == z) & (ds['N'] == n-1)].index[0]
            p3 = ds.loc[(ds['Z'] == z+1) & (ds['N'] == n)].index[0]
            n1 = ds.loc[(ds['Z'] == z+1) & (ds['N'] == n-2)].index[0]
            n2 = ds.loc[(ds['Z'] == z+2) & (ds['N'] == n-1)].index[0]
            GK_mat[i, p1], GK_mat[i, p2], GK_mat[i, p3] = (1,)*3
            GK_mat[i, i], GK_mat[i, n1], GK_mat[i, n2] = (-1,)*3
    except:
        pass


# def training step
def compute_training_step(optimizer):
    # predict
    pred = ann(data_x).flatten()

    # interpolation loss
    iloss = mse((pred[train_indexes]-train_f)/train_s, izeros) / ntrain

    # physics loss
    ploss = mse(torch.matmul(GK_mat, pred), zeros) / datalen

    # sum up
    loss = iloss + ploss * 1e10

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation loss
    vloss = mse((pred[val_indexes]-val_f)/val_s, vzeros) / nval

    return iloss, vloss


# run training
def train(epochs, lr=1e-2, quiet=False):
    data = []
    opt = RAdam(ann.parameters(), lr=lr)
    T0 = time()
    for step in range(1, epochs+1):
        t0 = time()
        tloss, vloss = compute_training_step(opt)
        data.append((step, tloss, vloss))
        tf = time()
        if not quiet:
            print(f"Epoch {step}/{epochs}\n\n",
                  f"Training loss {tloss}\n",
                  f"Validation loss {vloss}\n",
                  "Time per step: {:.6f} seg\n".format(tf-t0),
                  "Time spent: {:.6f} seg\n".format(tf-T0),
                  "-------------------\n\n")
    return pd.DataFrame(data=data, columns=['epochs', 'train loss', 'val loss'])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# do training
train_output = train(epochs=30000, lr=1e-5)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# wrap model
def ann_np(x):
    if isinstance(x, np.ndarray):
        if x.shape == (model_in, ):
            tensor_x = tensor(x, dtype=tfloat, device=device)
            return ann(tensor_x).detach().cpu().numpy()
        elif len(x[0]) == model_in and len(x) > 1:
            new_shape = (len(x), model_in)
            tensor_x = tensor(x.reshape(new_shape), dtype=tfloat, device=device)
            return ann(tensor_x).detach().cpu().numpy()
        else:
            print(x)
            raise Exception("can't compute this")
    else:
        raise Exception('input not a numpy array')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plots


# loss vs train plot
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(train_output['epochs'], train_output['train loss'], label='train loss')
ax.plot(train_output['epochs'], train_output['val loss'], label='val loss')
ax.legend(loc='upper right')
ax.set_yscale('log')

ax.tick_params(which='both', direction='in', top=True, right=True)
ax.tick_params(which='major', width=.8, length=4)
ax.tick_params(which='minor', width=.6, length=3)
ax.grid(color='grey', linestyle='-', linewidth=.25)
fig.tight_layout()
plt.show()


# results plot
preds = ann_np(ds.values[:, :-2]).flatten()
targets = ds.values[:, -2]
maxA, dm = targets.max(), 5
fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                         gridspec_kw={'width_ratios': [1, 1.2]})
# 1
ax = axes[0]
ax.scatter(targets[train_indexes], preds[train_indexes],
           color=yellow, label='Train set')
ax.scatter(targets[val_indexes], preds[val_indexes],
           color=green, label='Val set')
ax.plot([0, maxA], [0, maxA], color=red, label=f'+/-{dm} ua')
ax.plot([0, maxA], [dm, maxA+dm], color=red, lw=.5)
ax.plot([0, maxA], [-dm, maxA-dm], color=red, lw=.5)
ax.set_xlabel('m AME2020')
ax.set_ylabel('m NN')
ax.legend(loc='upper left')

# 2
errors = (preds - targets)/targets
dm = .001
ax = axes[1]
ax.scatter(ds['A'].values[train_indexes], errors[train_indexes],
           color=yellow, label='Train set')
ax.scatter(ds['A'].values[val_indexes], errors[val_indexes],
           color=green, label='Val set')
ax.plot([0, maxA], [0]*2, color=red, label=f'{dm*100}% relative error')
ax.plot([0, maxA], [dm]*2, color=red, lw=.5)
ax.plot([0, maxA], [-dm]*2, color=red, lw=.5)
ax.set_xlabel('A')
ax.legend()

for ax in axes:
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', width=.8, length=4)
    ax.tick_params(which='minor', width=.6, length=3)
    ax.grid(color='grey', linestyle='-', linewidth=.25)
fig.tight_layout()
plt.show()

# feature importance
cols = list(ds.columns)[:-2]
feats = ds[cols].values
explainer = shap.Explainer(ann_np, feats, feature_names=cols)
shap_values = explainer(feats)
shap.plots.beeswarm(shap_values, plot_size=(9, 5))
