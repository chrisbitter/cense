import shutil
import os

root_folder = "/home/cscheiderer/CENSE_gitlab/demonstrator_RLAlgorithm/GPU/"

if os.path.exists(root_folder + 'training_data'):
  print("Deleting folder 'training_data'")
  shutil.rmtree(root_folder + 'training_data')

print("Creating folder 'training_data/model'")
os.makedirs(root_folder + 'training_data/model')
print("Creating folder 'training_data/new_data'")
os.makedirs(root_folder + 'training_data/data/new_data')