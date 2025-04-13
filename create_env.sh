# conda is being super finnicky and I can't be asked so here is my alternative 
# to a conda yaml file
conda env create -n iconshop-new python=3.11 pip
conda activate iconshop-new
pip install torch torchaudio torchvision accelerate cairosvg einops ipython matplotlib networkx numpy pandas pillow scikit_learn scipy tensorboard tqdm transformers shapely