import synapseclient
import os


def main():
    print(f"current working directory: {os.getcwd()}")

    syn = synapseclient.Synapse()
    syn.login()

    # Obtain a pointer and download the data
    syn60086071 = syn.get(entity='syn60086071', downloadLocation='./data/BraTS/syn60086071.zip')

    # Get the path to the local copy of the data file
    filepath = syn60086071.path
    print(filepath)
    
   
if __name__ == '__main__':
    main()
