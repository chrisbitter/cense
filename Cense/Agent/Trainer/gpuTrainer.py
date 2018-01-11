# paramiko.util.log_to_file('/tmp/paramiko.log')
import json
import logging
import os

import h5py
import paramiko
import time

from threading import Thread


class GpuTrainer(object):

    done_training = True

    training_number = 0

    def __init__(self, trainer_config):

        self.epochs_start = trainer_config["epochs_start"]
        self.epochs_end = trainer_config["epochs_end"]
        self.batch_size_start = trainer_config["batch_size_start"]
        self.batch_size_end = trainer_config["batch_size_end"]
        self.trainings_before_param_update = trainer_config["trainings_before_param_update"]
        self.trainings_until_end_config = trainer_config["trainings_until_end_config"]
        self.discount_factor = trainer_config["discount_factor"]
        self.target_update_rate = trainer_config["target_update_rate"]

        self.current_gpu_config = {
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

        self.local_new_data = gpu_settings["local_data_root"] + gpu_settings["local_new_data"]
        self.local_model = gpu_settings["local_data_root"] + gpu_settings["local_model"]
        self.local_training_params = gpu_settings["local_data_root"] + gpu_settings["local_training_params"]

        self.remote_new_data = gpu_settings["remote_data_root"] + gpu_settings["remote_new_data"]
        self.remote_model = gpu_settings["remote_data_root"] + gpu_settings["remote_model"]
        self.remote_training_params = gpu_settings["remote_data_root"] + gpu_settings["remote_training_params"]

        self.remote_script_train = gpu_settings["remote_data_root"] + gpu_settings["remote_script_train"]
        self.remote_script_reset = gpu_settings["remote_data_root"] + gpu_settings["remote_script_reset"]

        self.remote_signal_train = gpu_settings["remote_data_root"] + gpu_settings["remote_signal_train"] + self.id
        self.remote_signal_alive = gpu_settings["remote_data_root"] + gpu_settings["remote_signal_alive"] + self.id

    def get_ssh_channel(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy())

        ssh.connect(self.host, self.port, self.username, self.password)

        return ssh

    def get_sftp_channel(self):
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        return transport, sftp

    def reset(self):
        # reset gpu

        ssh = self.get_ssh_channel()

        command = "python " + self.remote_script_reset

        stdin, stdout, stderr = ssh.exec_command(command)

        exit_status = stdout.channel.recv_exit_status()

        if exit_status != 0:
            print("Error: ", exit_status)
            [print(err) for err in stderr.readlines()]

        ssh.close()

    def train(self, states, actions, rewards, new_states, terminals):

        if self.host is None or self.port is None or self.username is None or self.password is None:
            print("Credentials missing!")
            return

        self.done_training = False
        self.training_number += 1

        config_changed = False

        if self.training_number <= self.trainings_until_end_config:
            if self.training_number % self.trainings_before_param_update == 0:
                self.current_gpu_config["epochs"] += \
                    (self.epochs_end - self.epochs_start) // self.trainings_until_end_config
                self.current_gpu_config["batch_size"] += \
                    (self.batch_size_end - self.batch_size_start) // self.trainings_until_end_config
                config_changed = True

        # if self.training_number == self.trainings_without_target + 1:
        #     self.current_gpu_config["use_target"] = 1
        #     config_changed = True

        # Send experience & configuration to GPU

        if os.path.isfile(self.local_new_data):
            os.remove(self.local_new_data)

        # pack data into hdf file (overwrites existing data!)
        with h5py.File(self.local_new_data, 'w') as f:
            f.create_dataset('states', data=states)
            f.create_dataset('actions', data=actions)
            f.create_dataset('rewards', data=rewards)
            f.create_dataset('new_states', data=new_states)
            f.create_dataset('terminals', data=terminals)

        gpu_alive = True

        try:
            transport, sftp = self.get_sftp_channel()

            sftp.stat(self.remote_signal_alive)

            transport.close()
            sftp.close()
        except IOError:
            gpu_alive = False

        command = "touch " + self.remote_signal_train

        ssh = self.get_ssh_channel()
        ssh.exec_command(command)
        ssh.close()

        # Upload Experience
        transport, sftp = self.get_sftp_channel()
        sftp.put(self.local_new_data, self.remote_new_data)

        # upload training parameters, if changed
        if config_changed:
            with open(self.local_training_params, 'w') as f:
                json.dump(self.current_gpu_config, f, sort_keys=True, indent=4)

            sftp.put(self.local_training_params, self.remote_training_params)

        transport.close()
        sftp.close()

        # if training file is dead, launch it (with id)
        if not gpu_alive:
            Thread(target=self.run_training_script).start()

        transport, sftp = self.get_sftp_channel()

        while True:
            try:
                sftp.stat(self.remote_signal_train)
            except IOError:
                # training_signal is gone, that means gpu is done training!
                break

        # Download Model
        sftp.get(self.remote_model, self.local_model)

        transport.close()
        sftp.close()

        self.done_training = True

    def run_training_script(self):
        print("run training script")

        logging.debug("Start script on GPU")
        if self.remote_script_train is None:
            raise IOError('script_remote missing!')

        command = "python " + self.remote_script_train + " " + self.id

        ssh = self.get_ssh_channel()

        #####
        stdin, stdout, stderr = ssh.exec_command(command)

        exit_status = stdout.channel.recv_exit_status()

        if exit_status != 0:
            print("Error: ", exit_status)
            [print(err) for err in stderr.readlines()]
        #####
        ssh.close()

    def send_model_to_gpu(self):

        # Upload Model
        transport, sftp = self.get_sftp_channel()
        sftp.put(self.local_model, self.remote_model)

        # upload initial config
        with open(self.local_training_params, 'w') as f:
            json.dump(self.current_gpu_config, f, sort_keys=True, indent=4)

        sftp.put(self.local_training_params, self.remote_training_params)
        transport.close()
        sftp.close()

    def is_done_training(self):
        return self.done_training


if __name__ == "__main__":
    gpu = GpuTrainer({
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
            "local_new_data": "new_data.h5",
            "model_config_local": "model.json",
            "model_weights_local": "weights.h5",
            "local_training_params": "train_params.json",

            "remote_data_root": "/home/useradmin/Dokumente/rm505424/CENSE/Christian/",
            "remote_new_data": "training_data/data/new_data/new_data.h5",
            "model_config_remote": "training_data/model/model.json",
            "model_weights_remote": "training_data/model/weights.h5",
            "remote_training_params": "training_data/train_params.json",

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
