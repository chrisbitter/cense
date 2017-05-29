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
    script_remote = None

    def __init__(self, project_root_folder):
        config_file = os.path.join(os.path.dirname(__file__), '..', 'Resources', 'my_file')

        with open(project_root_folder + 'Resources/credentials.json') as json_data:
            config = json.load(json_data)

        config = config["gpu"]
        # Open a transport

        self.host = config["host"]
        self.port = config["port"]

        self.username = config["user"]
        self.password = config["password"]

        self.new_data_local = config["local_data_root"] + config["new_data_local"]
        self.model_config_local = config["local_data_root"] + config["model_config_local"]
        self.model_weights_local = config["local_data_root"] + config["model_weights_local"]

        self.new_data_remote = config["remote_data_root"] + config["new_data_remote"]
        self.model_config_remote = config["remote_data_root"] + config["model_config_remote"]
        self.model_weights_remote = config["remote_data_root"] + config["model_weights_remote"]

        self.script_remote = config["remote_data_root"] + config["script_remote"]

        #todo test if configs are correct -> connect to gpu etc.

    def send_model_to_gpu(self):
        # init sftp
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Upload Model & Weights
        # todo: check if keras model save for lambda layers has been fixed. Until then, hardcode model on gpu!
        #sftp.put(self.model_config_local, self.model_config_remote)
        sftp.put(self.model_weights_local, self.model_weights_remote)

        # Close
        sftp.close()
        transport.close()

    def send_experience_to_gpu(self, states, actions, rewards, suc_states, terminals):

        #pack data into hdf file (overwrites existing data!)
        f = h5py.File(self.new_data_local, 'w')
        f.create_dataset('states', data=states)
        f.create_dataset('actions', data=actions)
        f.create_dataset('rewards', data=rewards)
        f.create_dataset('suc_states', data=suc_states)
        f.create_dataset('terminals', data=terminals)
        f.close()

        # init sftp
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Upload Experience
        logging.debug("local data", self.new_data_local)
        logging.debug("remote data", self.new_data_remote)

        sftp.put(self.new_data_local, self.new_data_remote)

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

    #
    # Downloads model weights from GPU
    # Returns path to model weights
    #
    def fetch_model_weights_from_gpu(self):
        if self.host is None or self.port is None or self.username is None or self.password is None:
            print("Credentials missing!")
            return

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

    def train_on_gpu(self):
        print("train on gpu")
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

        if exit_status == 0:
            print("Training on gpu done")
        else:
            print("Error: ", exit_status)
            [print(err) for err in stderr.readlines()]

        ssh.close()