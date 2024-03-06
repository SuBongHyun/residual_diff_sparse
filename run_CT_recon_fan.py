import os
import pdb

import torch
# from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import numpy as np
import controllable_generation_fan

from utils import restore_checkpoint, show_samples_gray, clear, clear_color, \
    lambda_schedule_const, lambda_schedule_linear
from pathlib import Path
from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import datasets
import time

from torch_radon import Radon, RadonFanbeam
import matplotlib.pyplot as plt

solver = 'POCS' #MCG or POCS
task = 'residual' #default or residual
config_name = 'AAPM_256_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000
ckpt_num = 108
N = num_scales

root = './samples/2016_AAPM'

# Parameters for the inverse problem
sparsity = 12
num_proj = 720 // sparsity
det_spacing = 1.5
sod = 595
dod = 595
size = 256
det_count = 512
schedule = 'linear'
start_lamb = 1.0
end_lamb = 0.6

num_posterior_sample = 1

if schedule == 'const':
    lamb_schedule = lambda_schedule_const(lamb=start_lamb)
elif schedule == 'linear':
    lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)
else:
    NotImplementedError(f"Given schedule {schedule} not implemented yet!")

freq = 1

if sde.lower() == 'vesde':
    from configs.ve import AAPM_256_ncsnpp_continuous as configs
    ckpt_filename = f"./checkpoints/{config_name}/checkpoint_{ckpt_num}.pth"
    config = configs.get_config()
    config.model.num_scales = N
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5

batch_size = 1
config.training.batch_size = batch_size
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False

snr = 0.16
lamb = 0.841
n_steps = 1

batch_size = 1
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)

state = dict(step=0, model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
ema.copy_to(score_model.parameters())

data_Num=len(os.listdir(Path(root)))
for i in range(data_Num):
    data_lst = os.listdir(Path(root))
    data_lst.sort()
    data_lst = data_lst[i]

    idx = int(data_lst.split('.')[0])

    filename = Path(root) / (data_lst)
    print(filename)
    save_root = Path(f'./results/SV-CT_fan/m{720/sparsity}/{idx}/{solver}')
    save_root.mkdir(parents=True, exist_ok=True)
    irl_types = ['input', 'recon', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)
    # Read data
    img = torch.from_numpy(np.load(filename))
    h, w = img.shape
    img = img.view(1, 1, h, w)
    img = img.to(config.device)
    ##
    plt.imsave(save_root / 'label' / f'{str(idx).zfill(4)}.png', clear(img), cmap='gray')
    # full
    angles = torch.FloatTensor(np.linspace(0, 2 * np.pi, num_proj,endpoint=False).astype(np.float32)).to(config.device)
    angles_all = torch.FloatTensor(np.linspace(0, 2 * np.pi, 720,endpoint=False).astype(np.float32)).to(config.device)
    radon = RadonFanbeam(resolution=size,angles=angles,source_distance=sod,det_distance=dod,det_spacing=det_spacing,clip_to_circle=False,det_count=det_count)
    radon_all = RadonFanbeam(resolution=size, angles=angles_all, source_distance=sod, det_distance=dod, det_spacing=det_spacing,
                          clip_to_circle=False, det_count=det_count)

    mask = torch.zeros([batch_size, 1, 720,det_count]).to(config.device)
    mask[:,:, ::sparsity,:] = 1

    sinogram = radon.forward(img)

    # FBP
    fbp = radon.backprojection(radon.filter_sinogram(sinogram))
    plt.imsave(str(save_root / 'input' / f'FBP.png'), clear(fbp), cmap='gray')
    print('%dth proj psnr_input: %f  ssim_input: %f' %(idx,psnr_input,ssim_input))
    print('%dth proj psnrC_input: %f  ssimC_input: %f' % (idx,psnrC_input, ssimC_input))
    print('solver: %s' % solver)
    print('task: %s' %task)

    pc_pocs = controllable_generation_fan.get_pc_radon_pocs(sde,
                                                      predictor, corrector,
                                                      inverse_scaler,
                                                      snr=snr,
                                                      n_steps=n_steps,
                                                      probability_flow=probability_flow,
                                                      continuous=config.training.continuous,
                                                      denoise=False,
                                                      radon=radon,
                                                      radon_all=radon_all,
                                                      weight=0.1,
                                                      save_progress=True,
                                                      save_root=save_root,
                                                      lamb_schedule=lamb_schedule,
                                                      mask=mask,task =task,solver=solver)
    x = pc_pocs(score_model, scaler(img), measurement=sinogram)

    plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}_{solver}_{task}_results.png'), clear(x), cmap='gray')


