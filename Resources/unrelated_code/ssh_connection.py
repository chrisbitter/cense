import paramiko
import json

with open('credentials.json') as json_data:
    config = json.load(json_data)

config = config["gpu"]

if __name__ == '__main__':

    host = config["host"]
    port = config["port"]
    username = config["user"]
    password = config["password"]

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(
        paramiko.AutoAddPolicy())

    ssh.connect(host, port, username, password)

    folder_on_gpu = config["folder_on_gpu"]

    config_path = "abc"
    new_data_path = "xyz"

    command = "python " + folder_on_gpu + "train_from_config.py " + config_path + " " + new_data_path
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.readlines())