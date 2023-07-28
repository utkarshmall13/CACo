conda create --name cacoenv python=3.9.13 -y
conda activate ssrep

conda install cudatoolkit=11.0 -y
python3 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install matplotlib pytorch-lightning==1.1.8 pytorch-lightning-bolts==0.3.0 scikit-learn rasterio lmdb pandas jupyter progressbar
python3 -m pip install wandb  opencv-python gym