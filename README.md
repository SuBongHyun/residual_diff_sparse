# Residual-Guidance Diffusion Model for Sparse-View CT Reconstuction  
![concept](./figs/workflow.jpg)

## Environment

After conda install, follow the ```install.sh``` script. we provide a torch environment. 
```bash
source install.sh
```
After then, install a [torch-radon](https://github.com/matteo-ronchetti/torch-radon) for fan-beam simulation. 
you can find other dependencies in  ```requirements.txt```.

## Training
1. If you use your own dataset for the training, you can set your data directory at './configs/ve/AAPM_256_ncsnpp_continuous.py'

2. Run the following command for the train.
```
sh train.sh
```

## Inference
1. If you use your own dataset for the sampling, please put them in './sample'

2. Run the following command for the test.
```
python run_CT_recon.py
```

Code is heavily based on [score-mri](https://github.com/HJ-harry/score-MRI) and [MCG](https://github.com/HJ-harry/MCG_diffusion). 
We highly appreciate your support for sharing code.
