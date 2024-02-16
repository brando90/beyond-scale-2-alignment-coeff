#!/bin/bash
# -- Check envs
conda info -e

# -- Activate conda
conda update -n base -c defaults conda
pip install --upgrade pip
conda create -y -n align_coeff python=3.10.11
conda activate align_coeff
#conda remove --name align_coeff --all

# -- Install this library from source
# - Get the code, put it in afs so its available to all machines and symlink it to home in the local machine
cd /afs/cs.stanford.edu/u/brando9/
git clone git@github.com:brando90/beyond-scale-2-alignment-coeff.git
ln -s /afs/cs.stanford.edu/u/brando9/beyond-scale-2-alignment-coeff $HOME/beyond-scale-2-alignment-coeff
# - Install the library in editable mode so that changes are reflected immediately in running code
pip install -e ~/beyond-scale-2-alignment-coeff
# pip uninstall ~/beyond-scale-2-alignment-coeff
cd ~/beyond-scale-2-alignment-coeff

# -- Test pytorch
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"
python -c "import torch; print(torch.version.cuda); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"
python -c "import torch; print(f'{torch.cuda.device_count()=}'); print(f'Device: {torch.cuda.get_device_name(0)=}')"

# -- Install uutils from source
# - Get the code, put it in afs so its available to all machines and symlink it to home in the local machine
cd /afs/cs.stanford.edu/u/brando9/
git clone git@github.com:brando90/ultimate-utils.git $HOME/ultimate-utils/
ln -s /afs/cs.stanford.edu/u/brando9/ultimate-utils $HOME/ultimate-utils
# - Install the library in editable mode so that changes are reflected immediately in running code
pip install -e ~/ultimate-utils
#pip uninstall ~/ultimate-utils

# -- Install ultimate-anatome
# - Get the code, put it in afs so its available to all machines and symlink it to home in the local machine
cd /afs/cs.stanford.edu/u/brando9/
git clone git@github.com:brando90/ultimate-anatome.git $HOME/ultimate-anatome/
ln -s /afs/cs.stanford.edu/u/brando9/ultimate-anatome $HOME/ultimate-anatome
# - Install the library in editable mode so that changes are reflected immediately in running code
pip install -e ~/ultimate-anatome
#pip uninstall ~/ultimate-anatome

# - Test uutils
python -c "import uutils; uutils.torch_uu.gpu_test()"

# -- Wandb
pip install wandb --upgrade
wandb login
#wandb login --relogin
cat ~/.netrc
