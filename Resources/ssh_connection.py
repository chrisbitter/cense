from pexpect import pxssh
import getpass


if __name__ == '__main__':
    try:
        s = pxssh.pxssh()
        hostname = 'nalaland.ddns.net'
        username = 'pi'
        password = 'Y#Rasany4$PiK'
        s.login(hostname, username, password)
        s.sendline('cd /home/pi/Thesis/data')  # run a command
        s.prompt()  # match the prompt
        print(s.before)  # print everything before the prompt.
        s.sendline('ls -l')
        s.prompt()
        print(s.before)
        s.logout()
    except pxssh.ExceptionPxssh as e:
        print("pxssh failed on login.")
        print(str(e))