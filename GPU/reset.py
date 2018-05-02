import shutil
import os.path as path
import os

def reset():
    root_folder = path.dirname(path.realpath(__file__))

    print(root_folder)

    if path.exists(path.join(root_folder, 'training_data')):
        print("Deleting folder 'training_data'")
        shutil.rmtree(path.join(root_folder, 'training_data'))

    print("Creating folder 'training_data/model'")
    os.makedirs(path.join(root_folder, 'training_data/model'))
    print("Creating folder 'training_data/new_data'")
    os.makedirs(path.join(root_folder, 'training_data/data/new_data'))

if __name__ == "__main__":
    reset()