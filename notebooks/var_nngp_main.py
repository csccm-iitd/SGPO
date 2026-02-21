# %%
import sys
sys.path.append('/home/user/Documents/VNN_GPO/src/')
import gpytorch
from gpytorch.means import MultitaskMean
from gpytorch.kernels import InducingPointKernel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from utilities3 import *
from pytorch_wavelets import DWT, IDWT
import scipy
from matplotlib.cm import ScalarMappable
from utils import *
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import MultitaskKernel, MaternKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import VariationalELBO
from gpytorch.variational.nearest_neighbor_variational_strategy import NNVariationalStrategy
from var_nngp import VNNGP
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.patches import Rectangle

# %%
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

# Disable cuDNN
# torch.backends.cudnn.enabled = False
print(f"device: {device}")
torch.manual_seed(100)
np.random.seed(100)


#%%
'''   
CONFIG AND DATA 
'''
ntrain = 1500
ntest = 100

step_size = 50
gamma = 0.75

lr = 1e-3
batch_size = 16
num_epochs = 150

r = 4 #15 # 5
h = int(((101 - 1)/r) + 1)
s = h

r1 = 4
h1 = int(((101 - 1)/r1) + 1)
s1=h1

PATH = '/home/user/Documents/GP_WNO/DATA/Darcy_Triangular_FNO.mat'

reader = MatReader(PATH)
x_train = reader.read_field('boundCoeff')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

x_test = reader.read_field('boundCoeff')[-ntest:,::r,::r][:,:s,:s]
y_test = reader.read_field('sol')[-ntest:,::r,::r][:,:s,:s]



#%%
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s1,s1,1)
x_test = x_test.reshape(ntest,s1,s1,1)

x_train = x_train[:ntrain,:]
y_train = y_train[:ntrain,:]
x_test = x_test[:ntest,:]
y_test = y_test[-ntest:,:]

x_tr = x_train.reshape(x_train.shape[0], -1)
y_tr = y_train.reshape(y_train.shape[0], -1)

# test data
x_t = x_test.reshape(x_test.shape[0], -1)
y_t = y_test.reshape(y_test.shape[0], -1)


x_tr = x_tr.to(device)
y_tr = y_tr.to(device)
x_t = x_t.to(device)
y_t = y_t.to(device)


train_dataset = torch.utils.data.TensorDataset(x_tr, y_tr)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=False)


#%%
num_latents = 120 #60 #140
num_tasks = y_tr.shape[-1]
num_nn = 90 #100 # number of nearest neighbor
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

model = VNNGP(x_train=x_tr,
                y_train=y_tr,
                likelihood=likelihood,
                num_nn=num_nn,
                num_latents= num_latents,
                training_batch_size=batch_size)

#%%
# Initialize the likelihood and VGP model
model = model.to(device)
likelihood = likelihood.to(device)
model.train()
likelihood.train()

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
mll = VariationalELBO(likelihood, model, num_data=y_tr.size(0))


''' 
TRAINING LOOP

'''
losses_nll = []
losses_mse = []
mse_loss = torch.nn.MSELoss()
rmse_loss = LpLoss(size_average=False)
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        # Forward pass
        output = model(batch_x)
        # print(output.shape)
        loss = -mll(output, batch_y)
        # Compute Mean Squared Error (MSE) loss
        loss_mse = mse_loss(output.mean, batch_y)
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        ls = model.covar_module.base_kernel.lengthscale.min().item() 
        outputscale = model.covar_module.outputscale.mean().item() 
        noise = model.likelihood.noise.mean().item()
    scheduler.step()
    losses_nll.append(loss.item())
    losses_mse.append(loss_mse.item())
    print(f'Epoch: {epoch+1}/{num_epochs} - elbo Loss: {loss.item():.3f} - MSE Loss: {loss_mse.item():.3f}')

#%%
'''
PREDICTION 
'''
# Create test DataLoader
test_dataset = TensorDataset(x_t, y_t)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
likelihood.eval()

all_means = []
all_variances = []


with torch.no_grad():
    for batch_x, _ in test_loader:  
        batch_x = batch_x.to(device)

        observed_pred = model(batch_x)

        mean_pred_batch = observed_pred.mean  
        var_pred_batch = observed_pred.variance  
        print(mean_pred_batch.shape)

        if mean_pred_batch.ndim == 3 and mean_pred_batch.shape[1] == 1:
            mean_pred_batch = mean_pred_batch.squeeze(1) 
        if var_pred_batch.ndim == 3 and var_pred_batch.shape[1] == 1:
            var_pred_batch = var_pred_batch.squeeze(1)

        # Append to the results
        all_means.append(mean_pred_batch.cpu().numpy())
        all_variances.append(var_pred_batch.cpu().numpy())

# Concatenate all batches
all_means_np = np.concatenate(all_means, axis=0)  # Shape: [ntest, dim]
all_variances_np = np.concatenate(all_variances, axis=0)  # Shape: [ntest, dim]

# Reshape and decode
mean_pred = torch.tensor(all_means_np)
mean_pred = mean_pred.reshape(mean_pred.shape[0], s,s)
mean_pred = y_normalizer.decode(mean_pred.detach().cpu())
mean_pred = mean_pred.reshape(mean_pred.shape[0], mean_pred.shape[1] * mean_pred.shape[2])

var_pred = torch.tensor(all_variances_np)
var_pred = var_pred.reshape(var_pred.shape[0], s, s)
var_pred = y_normalizer.decode(var_pred.detach().cpu())
var_pred = var_pred.reshape(var_pred.shape[0], var_pred.shape[1] * var_pred.shape[2])


#%%
'''
PREDICTION ERROR

'''
mse_loss = nn.MSELoss()
prediction_error = mse_loss(mean_pred.to(device), y_t)

relative_error = torch.mean(torch.linalg.norm(mean_pred.to(device)-y_t, axis = 1)/torch.linalg.norm(y_t, axis = 1))


print(f'Mean testing error: {(prediction_error).item()} ')
print(f'Mean relative error: {100*relative_error} % ')

#%%
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator, FormatStrFormatter

resl = 26

s = 1
xmax = s
ymax = s - 8 / 51

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 20

figure, axes = plt.subplots(1, 5, figsize=(24, 6))

index = 20

axes[0].set_title('B.C.'.format(index + 1), color='black', fontsize=28, fontweight='bold', pad=20)
image = axes[0].imshow(
    x_test[index, :, :, 0],
    cmap='seismic',
    extent=[0, 1, 0, 1],
    origin='lower',
    interpolation='Gaussian'
)
sm = ScalarMappable(cmap="seismic", norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
plt.colorbar(sm, ax=axes[0], fraction=0.045)
axes[0].set_ylabel("y", fontweight="bold", fontsize=28)
axes[0].set_xlabel("x", fontweight="bold", fontsize=28)
axes[0].tick_params(axis='both', labelsize="medium", width=2, direction='in')
axes[0].set_xticks([0, 0.5, 1])
axes[0].set_xticklabels(['0', '0.5', '1'])
axes[0].yaxis.set_major_locator(FixedLocator(axes[3].get_yticks()))
axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
    # label.set_fontweight('bold')

xf = np.array([0., xmax/2])
yf = xf * (ymax / (xmax/2))
axes[0].fill_between(xf, yf, ymax, color=[1, 1, 1])
xf = np.array([xmax/2, xmax])
yf = (xf - xmax) * (ymax / ((xmax/2) - xmax))
axes[0].fill_between(xf, yf, ymax, color=[1, 1, 1])
xf = np.array([0, xmax])
axes[0].fill_between(xf, ymax, s, color=[1, 1, 1])
axes[0].add_patch(Rectangle((0.5, 0), 0.01, 0.41, facecolor='white'))

# Truth
axes[1].set_title('Truth', color='black', fontsize=28, fontweight='bold', pad=20)
image = axes[1].imshow(
    y_t.detach().cpu().numpy().reshape(y_t.shape[0], resl, resl)[index, :, :],
    origin='lower',
    extent=[0, 1, 0, 1],
    interpolation='Gaussian',
    cmap='seismic'
)
sm = ScalarMappable(cmap="seismic", norm=plt.Normalize(vmin=-1.5, vmax=1))
sm.set_array([])
plt.colorbar(sm, ax=axes[1], fraction=0.045)
axes[1].set_xlabel("x", fontweight="bold",fontsize=28)
axes[1].tick_params(axis='both', labelsize="medium", width=2, direction='in')
axes[1].set_xticks([0, 0.5, 1])
axes[1].set_xticklabels(['0', '0.5', '1'])
axes[1].yaxis.set_major_locator(FixedLocator(axes[3].get_yticks()))
axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
    # label.set_fontweight('bold')
    label.set_fontsize(20)

xf = np.array([0., xmax/2])
yf = xf * (ymax / (xmax/2))
axes[1].fill_between(xf, yf, ymax, color=[1, 1, 1])
xf = np.array([xmax/2, xmax])
yf = (xf - xmax) * (ymax / ((xmax/2) - xmax))
axes[1].fill_between(xf, yf, ymax, color=[1, 1, 1])
xf = np.array([0, xmax])
axes[1].fill_between(xf, ymax, s, color=[1, 1, 1])
axes[1].add_patch(Rectangle((0.5, 0), 0.01, 0.41, facecolor='white'))

# Predictions
axes[2].set_title('Prediction', color='black', fontsize=28, fontweight='bold', pad=20)
image = axes[2].imshow(
    mean_pred.detach().cpu().numpy().reshape(mean_pred.shape[0], resl, resl)[index, :, :],
    origin='lower',
    extent=[0, 1, 0, 1],
    interpolation='Gaussian',
    cmap='seismic'
)
sm = ScalarMappable(cmap="seismic", norm=plt.Normalize(vmin=-1.5, vmax=1))
sm.set_array([])
plt.colorbar(sm, ax=axes[2], fraction=0.045)
axes[2].set_xlabel("x", fontweight="bold",fontsize=28)
axes[2].tick_params(axis='both', labelsize="medium", width=2, direction='in')
axes[2].set_xticks([0, 0.5, 1])
axes[2].set_xticklabels(['0', '0.5', '1'])
axes[2].yaxis.set_major_locator(FixedLocator(axes[3].get_yticks()))
axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
for label in axes[2].get_xticklabels() + axes[2].get_yticklabels():
    # label.set_fontweight('bold')
    label.set_fontsize(20)
xf = np.array([0., xmax/2])
yf = xf * (ymax / (xmax/2))
axes[2].fill_between(xf, yf, ymax, color=[1, 1, 1])
xf = np.array([xmax/2, xmax])
yf = (xf - xmax) * (ymax / ((xmax/2) - xmax))
axes[2].fill_between(xf, yf, ymax, color=[1, 1, 1])
xf = np.array([0, xmax])
axes[2].fill_between(xf, ymax, s, color=[1, 1, 1])
axes[2].add_patch(Rectangle((0.5, 0), 0.01, 0.41, facecolor='white'))

# Std
axes[3].set_title('Std', color='black', fontsize=28, fontweight='bold', pad=20)
image = axes[3].imshow(
    var_pred.detach().cpu().numpy().reshape(var_pred.shape[0], resl, resl)[index, :, :],
    cmap='seismic',
    extent=[0, 1, 0, 1],
    interpolation='Gaussian',
    origin='lower',
    # vmin=0.35,
    # vmax=0.95
)
sm = ScalarMappable(cmap="seismic", norm=plt.Normalize(vmin=1e-3, vmax=5e-2))
sm.set_array([])
plt.colorbar(sm, ax=axes[3], fraction=0.045, format='%.e')
# axes[3].set_ylabel('y', fontweight='bold',fontsize=24)
axes[3].set_xlabel("x", fontweight="bold",fontsize=28)
axes[3].tick_params(axis='both', labelsize="medium", width=2, direction='in')
axes[3].set_xticks([0, 0.5, 1])
axes[3].set_xticklabels(['0', '0.5', '1'])
axes[3].yaxis.set_major_locator(FixedLocator(axes[4].get_yticks()))
axes[3].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
for label in axes[3].get_xticklabels() + axes[3].get_yticklabels():
    # label.set_fontweight('bold')
    label.set_fontsize(20)

xf = np.array([0., xmax/2])
yf = xf * (ymax / (xmax/2))
axes[3].fill_between(xf, yf, ymax, color=[1, 1, 1])
xf = np.array([xmax/2, xmax])
yf = (xf - xmax) * (ymax / ((xmax/2) - xmax))
axes[3].fill_between(xf, yf, ymax, color=[1, 1, 1])
xf = np.array([0, xmax])
axes[3].fill_between(xf, ymax, s, color=[1, 1, 1])
axes[3].add_patch(Rectangle((0.5, 0), 0.01, 0.41, facecolor='white'))


# Error
axes[4].set_title('Error', color='black', fontsize=28, fontweight='bold', pad=20)
image = axes[4].imshow(
    np.abs(
        y_t.detach().cpu().numpy().reshape(y_t.shape[0], resl, resl)[index, :, :]
        - mean_pred.detach().cpu().numpy().reshape(mean_pred.shape[0], resl, resl)[index, :, :]
    ),
    cmap='jet',
    extent=[0, 1, 0, 1],
    interpolation='Gaussian',
    origin='lower',
    # vmax=2e-1,
    # vmin=3e-3
)
sm = ScalarMappable(cmap="seismic", norm=plt.Normalize(vmin=3e-2, vmax=2e-2))
sm.set_array([])
plt.colorbar(sm, ax=axes[4], fraction=0.045, format='%.e')
axes[4].set_xlabel("x", fontweight="bold",fontsize=28)
axes[4].tick_params(axis='both', labelsize="medium", width=2, direction='in')
axes[4].set_xticks([0, 0.5, 1])
axes[4].set_xticklabels(['0', '0.5', '1'])
axes[4].yaxis.set_major_locator(FixedLocator(axes[4].get_yticks()))
axes[4].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
for label in axes[4].get_xticklabels() + axes[4].get_yticklabels():
    # label.set_fontweight('bold')
    label.set_fontsize(20)

xf = np.array([0., xmax/2])
yf = xf * (ymax / (xmax/2))
axes[4].fill_between(xf, yf, ymax, color=[1, 1, 1])
xf = np.array([xmax/2, xmax])
yf = (xf - xmax) * (ymax / ((xmax/2) - xmax))
axes[4].fill_between(xf, yf, ymax, color=[1, 1, 1])
xf = np.array([0, xmax])
axes[4].fill_between(xf, ymax, s, color=[1, 1, 1])
axes[4].add_patch(Rectangle((0.5, 0), 0.01, 0.41, facecolor='white'))


plt.tight_layout()
# plt.savefig('/Darcy_notch/plots/SDD_Row_darcy_notch_01_3_6_24.pdf',format='pdf',dpi=600,bbox_inches='tight')


# %%

# %%
