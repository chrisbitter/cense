import paramiko
# paramiko.util.log_to_file('/tmp/paramiko.log')
import json

with open('credentials.json') as json_data:
    config = json.load(json_data)

config = config["gpu"]
# Open a transport

host = config["host"]
port = config["port"]
transport = paramiko.Transport((host, port))

# Auth
username = config["user"]
password = config["password"]
transport.connect(username=username, password=password)

# Go!

sftp = paramiko.SFTPClient.from_transport(transport)

# Download

# filepath = '/etc/passwd'
# localpath = '/home/remotepasswd'
# sftp.get(filepath, localpath)

# Upload

filepath = config["destination_file"]
localpath = config["local_file"]
sftp.put(localpath, filepath)

# Close

sftp.close()
transport.close()
