apt update
apt install openssh-server -y 
# service --status-all

apt install nano
nano /etc/ssh/sshd_config
# внутри файла sshd_config нужно поставить PermitRootLogin Yes
# PermitRootLogin prohibit-password
# 'PermitRootLogin Yes' 
# (для выхода из режима редактирования ctrl-X shift-Y Enter)
passwd root # обновить пароль для рута на '7042'

service ssh start
# после этого шага можно пробовать подключаться через putty
# ssh -oHostKeyAlgorithms=+ssh-rsa root@192.168.0.1