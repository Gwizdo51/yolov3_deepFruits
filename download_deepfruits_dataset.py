import os
import requests
from pathlib import Path
import argparse
from tqdm import tqdm


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 1000000

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-o', type=str, default='./', help='directory in which to store the zip file (default to same directory as script')

    args = parser.parse_args()

    zip_path = Path(args.output_dir) / "deepFruits_dataset.zip"

    if os.path.exists(zip_path):
        print("DeepFruits dataset already downloaded.")
    else:
        print("Downloading dataset zip file ...")
        download_file_from_google_drive("1dxZlIThKY7Lu-j9spZO8nP08s46Ufybv", zip_path)
        print("DeepFruits dataset downloaded.")