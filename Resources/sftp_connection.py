import paramiko
paramiko.util.log_to_file('/tmp/paramiko.log')

# Open a transport

host = "nalaland.ddns.net"
port = 22
transport = paramiko.Transport((host, port))

# Auth

username = "pi"
password = "Y#Rasany4$PiK"
transport.connect(username = username, password = password)

# Go!

sftp = paramiko.SFTPClient.from_transport(transport)

# Download

#filepath = '/etc/passwd'
#localpath = '/home/remotepasswd'
#sftp.get(filepath, localpath)

# Upload

filepath = '/home/pi/Thesis/data/Cense_wire_01.png'
localpath = '/home/chris/Documents/Cense_wire_01.png'
sftp.put(localpath, filepath)

# Close

sftp.close()
transport.close()