import pdb

from models import utils as mutils
import torch
import numpy as np
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools
from torch_radon import Radon, RadonFanbeam
from utils import show_samples, show_samples_gray, clear, clear_color
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
#import kornia

def get_pc_radon_pocs(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False, weight=1.0,
                     denoise=True, eps=1e-5, radon=None, radon_all=None, save_progress=False, save_root=None,
                     lamb_schedule=None, mask=None, measurement_noise=False,task='residual',solver='MCG'):
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def _A(x):
        return radon.forward(x)
    def _AT(sinogram): #backproj
        return radon.backprojection(sinogram)
    def _AINV(sinogram): #filtered-backproj
        return radon.backprojection(radon.filter_sinogram(sinogram))
    def _A_all(x):
        return radon_all.forward(x)
    def _AINV_all(sinogram):
        return radon_all.backprojection(radon_all.filter_sinogram(sinogram))

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, _, _ = update_fn(x, vec_t, model=model)
                return x

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None, i=None, norm_const=None):
            vec_t = torch.ones(data.shape[0], device=data.device) * t

            lamb = lamb_schedule.get_current_lambda(i)
            if solver == 'MCG':
                x = x.requires_grad_()
                x_next, x_next_mean, score = update_fn(x, vec_t,
                                                       model=model)
                _, bt = sde.marginal_prob(x, vec_t)
                hatx0 = x + (bt ** 2) * score

                diff= _AINV(measurement_clean - _A(hatx0))
                norm = torch.norm(diff)
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
                norm_grad *= weight
                norm_grad = _AINV_all(_A_all(norm_grad) * (1. - mask))
                _, std = sde.marginal_prob(measurement, vec_t)
                measurement_ = measurement + torch.rand_like(measurement) * std[:, None, None, None]
                x_next = x_next + lamb * _AT(measurement_ - _A(
                    x_next)) / norm_const - norm_grad
                if task=='residual':
                    res = _AINV(measurement_clean - _A(hatx0))
                    x_next = x_next + res *(lamb-0.6)*0.5
            else:
                x_next, x_next_mean, score = update_fn(x, vec_t,
                                                       model=model)
                _, bt = sde.marginal_prob(x, vec_t)
                hatx0 = x + (bt ** 2) * score
                _, std = sde.marginal_prob(measurement, vec_t)
                measurement_ = measurement_clean + torch.rand_like(measurement) * std[:, None, None, None]
                x_next = x_next + lamb * _AT(measurement_ - _A(
                    x_next)) / norm_const
                
                if task=='residual':
                    res = _AINV(measurement_clean - _A(hatx0))
                    x_next = x_next + res *(lamb-0.4) * 0.5 # for 15views: (lamb-0.6) better

            x_next = x_next.detach()
            return x_next

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, measurement=None):
        x = sde.prior_sampling(data.shape).to(data.device)

        ones = torch.ones_like(x).to(data.device)
        norm_const = _AT(_A(ones))
        timesteps = torch.linspace(sde.T, eps, sde.N)

        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            x = predictor_denoise_update_fn(model, data, x, t)
            x= corrector_radon_update_fn(model, data, x, t, measurement=measurement, i=i,
                                          norm_const=norm_const)
            if save_progress:
                if (i % 100) == 0:
                    plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x), cmap='gray')
        return inverse_scaler(x if denoise else x)

    return pc_radon
