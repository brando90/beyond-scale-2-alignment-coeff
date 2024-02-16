"""
https://github.com/alycialee/beyond-scale-language-data-diversity/tree/main/diversity#quick-start

conda install -c anaconda scikit-learn
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge tqdm
conda install -c conda-forge transformers
conda install -c conda-forge datasets


python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
python -c "import uutils; uutils.torch_uu.gpu_test()"

refs:
    - setup tools: https://setuptools.pypa.io/en/latest/userguide/package_discovery.html#using-find-or-find-packages
    - https://stackoverflow.com/questions/70295885/how-does-one-install-pytorch-and-related-tools-from-within-the-setup-py-install
"""
from setuptools import setup
from setuptools import find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='beyond-scale-2-alignment-coeff',  # project name
    version='0.0.1',
    description="Beyond Scale2: the Alignment Coefficient",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/brando90/beyond-scale-2-alignment-coeff',
    author='cBrando Miranda & Krrish Chawla',
    author_email='brando9@stanford.edu',
    python_requires='>=3.10.11',
    license='Apache 2.0',

    # currently
    package_dir={'': 'src'},
    packages=find_packages('src'),  # imports all modules/folders with  __init__.py & python files,
    # package_dir={'': 'ultimate-utils-proj-src'},
    # packages=find_packages('ultimate-utils-proj-src'),  # imports all modules/folders with  __init__.py & python files

    # for pytorch see doc string at the top of file
    install_requires=[
        'scikit-learn',
        'pandas',
        'numpy',
        'plotly',
        'wandb',
        'matplotlib',
        'seaborn',

        'torch',
        'torchvision',
        'torchaudio',
        'fairseq',
        
        'transformers',
        'datasets',

        'zstandard',  # needed for eval of all the pile

        'sentencepiece', # needed llama2 tokenizer

        'accelerate',

        'trl',
        'peft',

        'bitsandbytes',
        'einops',
    ]
)

