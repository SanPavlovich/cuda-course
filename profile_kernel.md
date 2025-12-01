# build image to prfile kernels!
https://leimao.github.io/blog/Docker-Nsight-Compute/

**build image:**
docker build -f nsight-compute.Dockerfile --no-cache --tag nsight-compute:12.4.1 .

**проброс GUI в докер контейнер на WINDOWS**
https://youtu.be/BDilFZ9C9mw

Еще нужно пробросить экран винды в контейнер через ssh, для этого нужно 2 программы:
https://youtu.be/FlHVuA_98SA - ссылка на видео с разбором этой херни
1) X Server, X Launch
2) Putty - пробросить X11 через ssh

- в самом контейнере нужно поднять ssh сервер!


- подкючение к докер контейнеру через putty:
host: root@192.168.0.1 port: 22
root@localhost

- в самом контейнере нужно набрать команду passwd и сменить пароль ('7042')
New password: 
Retype new password:
passwd: password updated successfully


Команда запуска ui: ncu-ui