# paramiko.util.log_to_file('/tmp/paramiko.log')
import json
import logging
import os

import h5py
import paramiko


class DummyTrainer(object):
    host = None
    port = None
    username = None
    password = None
    new_data_local = None
    new_data_remote = None

    model_config_local = None
    model_config_remote = None

    model_weights_local = None
    model_weights_remote = None

    training_params_local = None
    training_params_remote = None

    script_remote = None

    done_training = True

    training_number = 0

    def __init__(self, trainer_config, set_status_func):

        self.set_status_func = set_status_func
        set_status_func("Setup Trainer")

        self.epochs_start = trainer_config["epochs_start"]
        self.epochs_end = trainer_config["epochs_end"]
        self.batch_size_start = trainer_config["batch_size_start"]
        self.batch_size_end = trainer_config["batch_size_end"]
        self.trainings_before_param_update = trainer_config["trainings_before_param_update"]
        self.trainings_until_end_config = trainer_config["trainings_until_end_config"]
        self.trainings_without_target = trainer_config["trainings_without_target"]
        self.discount_factor = trainer_config["discount_factor"]
        self.target_update_rate = trainer_config["target_update_rate"]
        #self.buffer_size = trainer_config["buffer_size"]

        self.current_gpu_config = {
            "epochs": self.epochs_start,
            "batch_size": self.batch_size_start,
            "use_target": 0,
            "discount_factor": self.discount_factor,
            "target_update_rate": self.target_update_rate
            #"buffer_size": self.buffer_size
        }

        gpu_settings = trainer_config["gpu_settings"]
        # Open a transport

        self.host = gpu_settings["host"]
        self.port = gpu_settings["port"]

        self.username = gpu_settings["user"]
        self.password = gpu_settings["password"]

        self.new_data_local = gpu_settings["local_data_root"] + gpu_settings["new_data_local"]
        self.model_config_local = gpu_settings["local_data_root"] + gpu_settings["model_config_local"]
        self.model_weights_local = gpu_settings["local_data_root"] + gpu_settings["model_weights_local"]
        self.training_params_local = gpu_settings["local_data_root"] + gpu_settings["training_params_local"]

        self.new_data_remote = gpu_settings["remote_data_root"] + gpu_settings["new_data_remote"]
        self.model_config_remote = gpu_settings["remote_data_root"] + gpu_settings["model_config_remote"]
        self.model_weights_remote = gpu_settings["remote_data_root"] + gpu_settings["model_weights_remote"]
        self.training_params_remote = gpu_settings["remote_data_root"] + gpu_settings["training_params_remote"]

        self.script_remote = gpu_settings["remote_data_root"] + gpu_settings["script_remote"]
        self.test_script_remote = gpu_settings["remote_data_root"] + gpu_settings["test_script_remote"]
        self.script_reset = gpu_settings["remote_data_root"] + gpu_settings["script_reset"]

    def reset(self):
        pass

    def train(self, states, actions, rewards, suc_states, terminals, velocities=None, suc_velocities=None):
        pass

    def send_model_to_gpu(self):
        pass

    #
    # Downloads model weights from GPU
    # Returns path to model weights
    #
    def fetch_model_config_from_gpu(self):
        pass

    def test_on_gpu(self):
        pass

    def is_done_training(self):
        return self.done_training


if __name__ == "__main__":
    gpu = DummyTrainer({
        "epochs_start": 500,
        "epochs_end": 1000,
        "batch_size_start": 25,
        "batch_size_end": 100,
        "trainings_before_param_update": 5,
        "trainings_until_end_config": 25,
        "trainings_without_target": 3,
        "discount_factor": 0.99,
        "target_update_rate": 0.01,
        "gpu_settings": {
            "host": "137.226.189.187",
            "port": 22,
            "user": "useradmin",
            "password": "cocacola",

            "local_data_root": "C:\\Users\\Christian\\Thesis\\workspace\\CENSE\\demonstrator_RLAlgorithm\\Resources\\nn-data\\",
            "new_data_local": "new_data.h5",
            "model_config_local": "model.json",
            "model_weights_local": "weights.h5",
            "training_params_local": "train_params.json",

            "remote_data_root": "/home/useradmin/Dokumente/rm505424/CENSE/Christian/",
            "new_data_remote": "training_data/data/new_data/new_data.h5",
            "model_config_remote": "training_data/model/model.json",
            "model_weights_remote": "training_data/model/weights.h5",
            "training_params_remote": "training_data/train_params.json",

            "script_remote": "train_model_acceleration.py",
            "test_script_remote": "test_model.py",
            "script_reset": "reset.py"
        }
    }, print)

    gpu.reset()

    # import Cense.Agent.NeuralNetworkFactory.nnFactory as Factory
    # import numpy as np
    #
    # model = Factory.model_simple_conv((50, 50), 6)
    # print("local", model.predict(np.ones((1, 50, 50)), batch_size=1))
    #
    # model.save_weights(gpu.model_weights_local)
    # gpu.send_model_to_gpu()
    #
    # gpu.test_on_gpu()
    #
    # model.load_weights(gpu.model_weights_local)
    #
    # print("local", model.predict(np.ones((1, 50, 50)), batch_size=1))
    #
    print("done")
