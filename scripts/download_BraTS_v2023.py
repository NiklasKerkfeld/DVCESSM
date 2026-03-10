from pathlib import Path
import os
import synapseclient
import zipfile

def main():
    print(f"current working directory: {os.getcwd()}")

    syn = synapseclient.Synapse()
    syn.login()

    filepath = Path('./data/BraTS/syn60086071.zip')
    if not filepath.is_file():
        syn60086071 = syn.get(entity='syn60086071', downloadLocation='./data/BraTS/syn60086071.zip')

        print(f"Download complete!")

    if not os.path.isdir('./data/BraTS/V2023'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall('./data/BraTS/V2023')

        print("Extraction complete!")


if __name__ == '__main__':
    main()
