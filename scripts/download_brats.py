import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download(dataset='dschettler8845/brats-2021-task1', dest='data'):
    api = KaggleApi()
    api.authenticate()
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {dataset} to {dest} …")
    api.dataset_download_files(dataset, path=dest, unzip=True, quiet=False)
    print("Done!")

if __name__ == '__main__':
    download()