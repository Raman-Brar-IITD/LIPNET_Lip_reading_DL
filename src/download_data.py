import os
import gdown
import yaml

def download_data():
    """Downloads the dataset zip file."""
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    data_params = params['data']
    raw_dir = data_params['raw_dir']
    zip_path = os.path.join(raw_dir, data_params['zip_file'])
    data_url = data_params['url']

    os.makedirs(raw_dir, exist_ok=True)

    if not os.path.exists(zip_path):
        print(f"Downloading data from {data_url} to {zip_path}...")
        gdown.download(data_url, zip_path, quiet=False)
        print("Download complete.")
    else:
        print(f"Data archive {zip_path} already exists. Skipping download.")

if __name__ == '__main__':
    download_data()
