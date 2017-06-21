import paramiko
import h5py
# paramiko.util.log_to_file('/tmp/paramiko.log')
import json
import os
import logging

class GPU_Trainer(object):

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

    def __init__(self, project_root_folder, trainer_config):
        print("Setup Trainer")

        self.epochs_start = trainer_config["epochs_start"]
        self.epochs_end = trainer_config["epochs_end"]
        self.batch_size_start = trainer_config["batch_size_start"]
        self.batch_size_end = trainer_config["batch_size_end"]
        self.trainings_before_param_update = trainer_config["trainings_before_param_update"]
        self.trainings_until_end_config = trainer_config["trainings_until_end_config"]
        self.trainings_without_target = trainer_config["trainings_without_target"]
        self.discount_factor = trainer_config["discount_factor"]
        self.target_update_rate = trainer_config["target_update_rate"]

        self.current_gpu_config = {
            "epochs": self.epochs_start,
            "batch_size": self.batch_size_start,
            "use_target": 0,
            "discount_factor": self.discount_factor,
            "target_update_rate": self.target_update_rate
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
        #todo test if configs are correct -> connect to gpu etc.

    def train(self, experience):

        if self.host is None or self.port is None or self.username is None or self.password is None:
            print("Credentials missing!")
            return

        self.done_training = False
        self.training_number += 1

        config_changed = False

        if self.training_number % self.trainings_before_param_update == 0:
            self.current_gpu_config["epochs"] += (self.epochs_end - self.epochs_start) // self.trainings_until_end_config
            self.current_gpu_config["batch_size"] += (self.batch_size_end - self.batch_size_start) // self.trainings_until_end_config
            config_changed = True

        if self.training_number == self.trainings_without_target + 1:
            self.current_gpu_config["use_target"] = 1
            config_changed = True

        # Send experience & configuration to GPU
        # pack data into hdf file (overwrites existing data!)
        with h5py.File(self.new_data_local, 'w') as f:
            f.create_dataset('states', data=experience[0])
            f.create_dataset('actions', data=experience[1])
            f.create_dataset('rewards', data=experience[2])
            f.create_dataset('suc_states', data=experience[3])
            f.create_dataset('terminals', data=experience[4])

        # init sftp
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Upload Experience
        sftp.put(self.new_data_local, self.new_data_remote)

        # upload config, if changed
        if config_changed:
            with open(self.training_params_local, 'w') as f:
                json.dump(self.current_gpu_config, f, sort_keys=True, indent=4)

            sftp.put(self.training_params_local, self.training_params_remote)

        # Close
        sftp.close()
        transport.close()

        logging.debug("Training on GPU")
        if self.script_remote is None:
            print("script_remote missing!")
            return None

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy())

        ssh.connect(self.host, self.port, self.username, self.password)

        command = "python " + self.script_remote
        stdin, stdout, stderr = ssh.exec_command(command)

        exit_status = stdout.channel.recv_exit_status()

        if exit_status != 0:
            print("Error: ", exit_status)
            [print(err) for err in stderr.readlines()]

        ssh.close()

        # download the new weights

        # init sftp
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Download Model
        logging.debug("Weights remote: ", self.model_weights_remote)
        logging.debug("Weights local: ", self.model_weights_local)
        sftp.get(self.model_weights_remote, self.model_weights_local)

        # Close
        sftp.close()
        transport.close()

        self.done_training = True

    def send_model_to_gpu(self):
        # init sftp
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Upload Model & Weights
        # todo: check if keras model save for lambda layers has been fixed. Until then, hardcode model on gpu!
        #sftp.put(self.model_config_local, self.model_config_remote)
        sftp.put(self.model_weights_local, self.model_weights_remote)

        # upload initial config
        with open(self.training_params_local, 'w') as f:
            json.dump(self.current_gpu_config, f, sort_keys=True, indent=4)

        sftp.put(self.training_params_local, self.training_params_remote)

        # Close
        sftp.close()
        transport.close()

    #
    # Downloads model weights from GPU
    # Returns path to model weights
    #
    def fetch_model_config_from_gpu(self):
        # init sftp
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Download Model
        sftp.get(self.model_config_remote, self.model_config_local)

        # Close
        sftp.close()
        transport.close()

        return self.model_config_local

    def test_on_gpu(self):
        print("test on gpu")
        if self.test_script_remote is None:
            print("test_script_remote missing!")
            return None

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy())

        ssh.connect(self.host, self.port, self.username, self.password)

        command = "python " + self.test_script_remote
        stdin, stdout, stderr = ssh.exec_command(command)

        exit_status = stdout.channel.recv_exit_status()

        if exit_status != 0:
            print("Error: ", exit_status)
            [print(err) for err in stderr.readlines()]

        print(stdout.read())

        ssh.close()

    def is_done_training(self):
        return self.done_training

if __name__ == "__main__":

    gpu = GPU_Trainer(os.path.join(os.getcwd(), "..", "..", ""))

    import Cense.NeuralNetworkFactory.nnFactory as Factory
    import numpy as np

    model = Factory.model_simple_conv((50,50), 6)
    print("local", model.predict(np.ones((1, 50, 50)), batch_size=1))

    model.save_weights(gpu.model_weights_local)
    gpu.send_model_to_gpu()

    gpu.test_on_gpu()

    model.load_weights(gpu.model_weights_local)

    print("local", model.predict(np.ones((1, 50, 50)), batch_size=1))

    print("done")


