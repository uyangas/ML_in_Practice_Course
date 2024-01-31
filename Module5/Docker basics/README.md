# Docker ашиглах нь

Docker-н түгээмэл коммандууд

1. `docker build -t <image_name>:<tag>.`
1. `docker run -dp 8081:5002 -ti --name <container_name> <image_name>:<tag>`
1. `docker images`
1. `docker rmi <image_id or image_name>`
1. `docker ps`
1. `docker stop <container_id or container_name>`
1. `docker rm <container_id or contaienr_name>`
1. `docker pull <image_name>:<tag>`
1. `docker push <image_name>:<tag>`
1. `docker login -u "username" -p "password"`