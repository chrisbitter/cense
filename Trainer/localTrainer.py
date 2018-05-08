# paramiko.util.log_to_file('/tmp/paramiko.log')
import json
import logging
import os.path as path
import os

import h5py
import paramiko
import time

from threading import Thread
from shutil import copyfile
from pathlib import Path
from GPU.train_ac import train
from GPU.reset import reset

class LocalTrainer(object):
    done_training = True

    training_number = 0

    def __init__(self, project_root, trainer_config):

        self.epochs_start = trainer_config["epochs_start"]
        self.epochs_end = trainer_config["epochs_end"]
        self.batch_size_start = trainer_config["batch_size_start"]
        self.batch_size_end = trainer_config["batch_size_end"]
        self.trainings_before_param_update = trainer_config["trainings_before_param_update"]
        self.trainings_until_end_config = trainer_config["trainings_until_end_config"]
        self.discount_factor = trainer_config["discount_factor"]
        self.target_update_rate = trainer_config["target_update_rate"]

        self.current_train_params = {
            "epochs": self.epochs_start,
            "batch_size": self.batch_size_start,
            "discount_factor": self.discount_factor,
            "target_update_rate": self.target_update_rate
        }

        gpu_settings = trainer_config["gpu_settings"]

        # Open a transport
        self.host = gpu_settings["host"]
        self.port = gpu_settings["port"]

        self.username = gpu_settings["user"]
        self.password = gpu_settings["password"]

        self.id = time.strftime('%Y%m%d-%H%M%S')

        local_data_location = path.abspath(path.join(project_root, gpu_settings["local_data_root"]))

        self.local_new_data = path.abspath(path.join(local_data_location, gpu_settings["local_data"]))
        self.local_model = path.abspath(path.join(local_data_location, gpu_settings["local_model"]))
        self.local_training_params = path.abspath(path.join(local_data_location, gpu_settings["local_training_params"]))

        gpu_root = path.abspath(path.join(project_root, "GPU"))

        self.remote_new_data = path.join(gpu_root, gpu_settings["remote_data"])
        self.remote_model = path.join(gpu_root, gpu_settings["remote_model"])
        self.remote_training_params = path.join(gpu_root, gpu_settings["remote_training_params"])

        self.remote_script_train = path.join(gpu_root, gpu_settings["remote_script_train"])
        self.remote_script_reset = path.join(gpu_root, gpu_settings["remote_script_reset"])

        self.remote_signal_train = path.join(gpu_root, gpu_settings["remote_signal_train"] + self.id)
        self.remote_signal_alive = path.join(gpu_root, gpu_settings["remote_signal_alive"] + self.id)

    def reset(self):
        # reset gpu
        reset()

    def train(self, states, actions, rewards, new_states, terminals, graph):

        self.done_training = False
        self.training_number += 1

        config_changed = False

        if self.training_number <= self.trainings_until_end_config:
            if self.training_number % self.trainings_before_param_update == 0:
                self.current_train_params["epochs"] += \
                    (self.epochs_end - self.epochs_start) // self.trainings_until_end_config
                self.current_train_params["batch_size"] += \
                    (self.batch_size_end - self.batch_size_start) // self.trainings_until_end_config
                config_changed = True

        # Send experience & configuration to GPU

        if path.isfile(self.local_new_data):
            os.remove(self.local_new_data)

        # pack data into hdf file (overwrites existing data!)
        with h5py.File(self.local_new_data, 'w') as f:
            f.create_dataset('states', data=states)
            f.create_dataset('actions', data=actions)
            f.create_dataset('rewards', data=rewards)
            f.create_dataset('new_states', data=new_states)
            f.create_dataset('terminals', data=terminals)

        gpu_alive = True

        if not path.exists(self.remote_signal_alive):
            gpu_alive = False

        Path(self.remote_signal_train).touch()

        # Upload Experience
        copyfile(self.local_new_data, self.remote_new_data)

        # upload training parameters, if changed
        if config_changed:
            with open(self.local_training_params, 'w') as f:
                json.dump(self.current_train_params, f, sort_keys=True, indent=4)

            copyfile(self.local_training_params, self.remote_training_params)

        # if training file is dead, launch it (with id)
        if not gpu_alive:
            Thread(target=self.run_training_script, args=(graph)).start()

        while path.exists(self.remote_signal_train):
            pass

        # Download Model
        copyfile(self.remote_model, self.local_model)

        self.done_training = True

    def run_training_script(self, graph):
        print("run training script")

        train(self.id, graph)

    def send_model_to_gpu(self):

        print(self.local_model, self.remote_model)

        copyfile(self.local_model, self.remote_model)

        # upload initial config
        with open(self.local_training_params, 'w') as f:
            json.dump(self.current_train_params, f, sort_keys=True, indent=4)

        copyfile(self.local_training_params, self.remote_training_params)

    def is_done_training(self):
        return self.done_training


if __name__ == "__main__":
    gpu = LocalTrainer(r"C:\Gitlab\demonstrator_RLAlgorithm",
                       {
                           "epochs_start": 1,
                           "epochs_end": 1,
                           "batch_size_start": 32,
                           "batch_size_end": 32,
                           "trainings_before_param_update": 1,
                           "trainings_until_end_config": 1,
                           "discount_factor": 0.99,
                           "target_update_rate": 0.001,
                           "gpu_settings": {
                               "host": "192.168.1.20",
                               "port": 22,
                               "user": "cscheiderer",
                               "password": "universalrobot5",

                               "local_data_root": "Resources/nn-data",
                               "local_data": "data.h5",
                               "local_model": "actor.h5",
                               "local_training_params": "train_params.json",

                               "remote_data_root": "GPU",
                               "remote_data": "training_data/data/new_data/data.h5",
                               "remote_model": "training_data/model/actor.h5",
                               "remote_training_params": "training_data/train_params.json",

                               "remote_signal_train": "training_data/training_signal_",
                               "remote_signal_alive": "training_data/alive_signal_",

                               "remote_script_train": "train_ac.py",
                               "remote_script_reset": "reset.py"
                           }
                       })

    gpu.reset()
    print("done")
