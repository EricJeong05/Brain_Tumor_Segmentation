@echo off
REM Install dependencies for BraTS project
pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --extra-index-url https://download.pytorch.org/whl/cu129
pip install "monai[itk,nibabel,scikit-image]==1.4.0"
pip install pytorch-ignite nibabel scikit-image scipy pillow tensorboard gdown tqdm lmdb psutil pandas einops transformers mlflow pynrrd clearml
