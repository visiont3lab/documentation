---
id: installation_and_samples_scripts
title: Installation Scripts 
sidebar_label: Installation Scripts  
custom_edit_url: https://github.com/visiont3lab/documentation/edit/master/docs/installation_and_samples_scripts.md
---


## Installation Scripts Opencv 2.4 for Ubuntu 18.04 LTS

Opencv 2.4 installation with static library. No cuda support.

```
#!/bin/bash  
echo "------------------------------------------------"
echo "Welcome to the Installation script, let's start!"
echo "------------------------------------------------"

echo "------------------------------------------------"
echo "Install OpenCV"
echo "------------------------------------------------"

sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt -y update
sudo apt -y install libjasper1 
sudo apt-get install openexr

sudo apt-get install libgtk2.0-dev 
mkdir -p $HOME/lib
cd  $HOME/lib
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 2.4
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_opencv_flann=OFF
 -D BUILD_opencv_photo=OFF -D BUILD_opencv_ml=OFF -D BUILD_opencv_objdetect=OFF -D BUILD_opencv_video=OFF -D BUILD_opencv_superres=OFF
  -D BUILD_SHARED_LIBS=OFF -D WITH_CUDA=OFF -D WITH_GTK=ON -D WITH_TIFF=OFF ..

echo "------------------------------------------------"
echo "Installation completed"
echo "------------------------------------------------"
```


## Installation Scripts Opencv 4 for Ubuntu 18.04 LTS

Installation Opencv 4 with Cuda Support. [Reference page](https://www.learnopencv.com/install-opencv-4-on-ubuntu-18-04/)

```
#!/bin/bash  
echo "------------------------------------------------"
echo "Welcome to the Installation script, let's start!"
echo "------------------------------------------------"

echo "------------------------------------------------"
echo "Install OpenCV 4 compile with Cuda"
echo "------------------------------------------------"

# Installation steps

# Compulsory dependencies
sudo apt -y install build-essential checkinstall cmake pkg-config yasm
sudo apt -y install git gfortran
sudo apt -y install libjpeg8-dev libpng-dev
sudo apt -y install software-properties-common
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt -y update
sudo apt -y install libjasper1 # This dependency is available in xenial repo
sudo apt -y install libtiff-dev
sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt -y install libxine2-dev libv4l-dev
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd "$cwd" # Not clear what it does
sudo apt -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt -y install libgtk2.0-dev libtbb-dev qt5-default
sudo apt -y install libatlas-base-dev
sudo apt -y install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt -y install libvorbis-dev libxvidcore-dev
sudo apt -y install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt -y install libavresample-dev
sudo apt -y install x264 v4l-utils
# Optional dependencies
sudo apt -y install libprotobuf-dev protobuf-compiler
sudo apt -y install libgoogle-glog-dev libgflags-dev
sudo apt -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

# Setup opencv and compilation
mkdir -p $HOME/lib && cd $HOME/lib
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout master #  4.0.1-dev Commit 4.0.1-66-ge5917a8fa 
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_GENERATE_PKGCONFIG=ON -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_PNG=OFF -DBUILD_TIFF=OFF -DBUILD_TBB=OFF -DBUILD_JPEG=OFF \
 -DBUILD_JASPER=OFF -DBUILD_ZLIB=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=ON -DBUILD_opencv_python3=ON -DWITH_V4L=ON -DWITH_OPENGL=ON \
  -DWITH_OPENCL=OFF -DWITH_OPENMP=OFF -DWITH_FFMPEG=ON -DWITH_GSTREAMER=OFF -DWITH_GSTREAMER_0_10=OFF -DWITH_CUDA=ON -DWITH_NVCUVID=OFF -DWITH_GTK=ON -DWITH_VTK=OFF \
  -DWITH_TBB=ON -DWITH_1394=OFF -DWITH_OPENEXR=OFF -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DCUDA_ARCH_BIN='5.0 6.0 6.1 6.2' \
  -DCUDA_ARCH_PTX="" -DINSTALL_C_EXAMPLES=OFF -DINSTALL_TESTS=OFF -DINSTALL_PYTHON_EXAMPLES=OFF ..
make
# sudo make install

echo "------------------------------------------------"
echo "Installation completed"
echo "------------------------------------------------"

# ----------------------------- Extra  information
# To solve opencv unsupported GNU version! gcc versions later than 6 are not supported!
# ADD -DCMAKE_C_COMPILER=/usr/bin/gcc-6  , note that it is not imporntant to set also g++ ( I am using g++ 7.3)
# Cjeck your GPU Architecture on https://en.wikipedia.org/wiki/CUDA
# DO not worry about sudo make install (the installation prefix /usr/local will automatically override https://stackoverflow.com/questions/9276169/removing-all-installed-opencv-libs
# To use the opencv installed library oin python do
# $ workon python3-env 
# $ pip uninstall opencv
# The compilation will automaticall generate the package config file "opencv.pc" inside /usr/local/lib 
# $ cp $THERMOHUMAN/lib/opencv/release/lib/python3/cv2.cpython-36m-x86_64-linux-gnu.so $HOME/.virtualenvs/python3-env/lib/python3.6/site-packages/cv2.so
# $ python -c "import cv2; print(cv2.__version__); print(cv2.getBuildInformation())"
```
 
 ## Installation Script to Build tensorflow from Source

 Installation script to build Tensorflow "libtensorflow_cc.so" from Source.

```
#!/bin/bash  

# Main reference https://medium.com/@Oysiyl/install-tensorflow-1-8-0-with-gpu-from-source-on-ubuntu-18-04-bionic-beaver-35cfa9df3600
# Second reference  https://gist.github.com/Brainiarc7/6d6c3f23ea057775b72c52817759b25c
# Third  reference https://medium.com/@fanzongshaoxing/use-tensorflow-c-api-with-opencv3-bacb83ca5683
echo "------------------------------------------------"
echo "Welcome to the Installation script, let's start!"
echo "------------------------------------------------"

#sudo apt-get install cmake          # Compiler dependency

echo "------------------------------------------------"
echo "Install Bazel  0.14.0"
echo "------------------------------------------------"

# Reference: https://docs.bazel.build/versions/master/install-ubuntu.html
# Issue with this cause the  fact that bazel update faste than tensolfow
mkdir -p $HOME/lib && cd $HOME/lib
sudo apt-get install openjdk-8-jdk
wget https://github.com/bazelbuild/bazel/releases/download/0.14.0/bazel-0.14.0-installer-linux-x86_64.sh
chmod +x bazel-0.14.0-installer-linux-x86_64.sh
./bazel-0.14.0-installer-linux-x86_64.sh --user
echo 'export PATH="$PATH:$HOME/bin"' >> ~/.bashrc


echo "------------------------------------------------"
echo "Install Tensorflow"
echo "------------------------------------------------"
# Reference:  https://tuanphuc.github.io/standalone-tensorflow-cpp/
sudo apt-get install python-numpy python-dev python-pip python-wheel 
cd $HOME/lib
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout r1.8
git pull
./configure
bazel build  --jobs 2 -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=monolithic //tensorflow:libtensorflow_cc.so
echo "------------------------------------------------"
echo "Installation completed"
echo "------------------------------------------------"
```

```
# ---------- Issue
# Bazel Issue
# Install latest bazel 25 
sudo apt-get install openjdk-8-jdk  
cd $HOME/lib
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel  # Install bazel 0.25.2 (check $ bazel version) 

# Absl Issue
# if u have this problem 
# at this commit 12e86468e2a9b57e636c1d0afcdf3f657f6df0b6
# we have absl issue. https://github.com/tensorflow/tensorflow/issues/22007
# It is solved doing
cd thermohuman_ws/lib/tensorflow
git clone https://github.com/abseil/abseil-cpp.git
ln -s abseil-cpp/absl ./absl
```

```
./configure
You have bazel 0.14.0 installed.
Please specify the location of python. [Default is /usr/bin/python]: 

Found possible Python library paths:
  /usr/local/lib/python2.7/dist-packages
  /usr/lib/python2.7/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: y
jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
No Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
	--config=mkl         	# Build with MKL support.
	--config=monolithic  	# Config for mostly static monolithic build.
Configuration finished
```

## Installation to build SimpleElastix Lib

Medical Image Registration Library. [Reference](https://simpleelastix.github.io/)

```
#!/bin/bash  
echo "------------------------------------------------"
echo "Welcome to the Installation script, let's start!"
echo "------------------------------------------------"

echo "------------------------------------------------"
echo "Install SimpleElastix"
echo "------------------------------------------------"

cd $HOME/lib
git clone https://github.com/SuperElastix/SimpleElastix
cd SimpleElastix
mkdir build
cd build
cmake ../SuperBuild
make -j2   #(j2 Means number of cores. if u have 16GB of ram u can run make -j4 otherwise if u have 8 u have to run make -j2)
cd $THERMOHUMAN/lib/SimpleElastix/build/SimpleITK-build/Wrapping/Python/Packaging
sudo python setup.py install

echo "------------------------------------------------"
echo "Installation completed"
echo "------------------------------------------------"


# Might be useful: cmake -D BUILD_EXAMPLES=OFF -D BUILD_TESTING=OFF -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=$HOME/SimpleElastix/build -D WRAP_CSHARP=OFF -D  WRAP_JAVA=OFF -D WRAP_LUA=OFF -D WRAP_PYTHON=OFF -D WRAP_R=OFF -D WRAP_RUBY=OFF -D WRAP_TCL=OFF ..

# Cmake is required if u have error do
# https://medium.com/@fanzongshaoxing/use-tensorflow-c-api-with-opencv3-bacb83ca5683
# Dowload cmake version 1.1 form here https://github.com/Kitware/CMake/releases
# tar xcvf cmake-3.11.1-Linux-x86_64.tar.gz
# cd cmake-3.14.4-Linux-x86_64
# sudo apt-get purge cmake
# sudo cp -r bin /usr/
# sudo cp -r share /usr/
# sudo cp -r doc /usr/share/
# sudo cp -r man /usr/share/
```

## Installation script Qt creator

```
#!/bin/bash 
echo "------------------------------------------------"
echo "-------------- Qt Creator 3.5.1 ----------------"
#https://www.lucidarme.me/how-install-documentation-for-qt-creator/
sudo apt-get install build-essential
sudo apt-get install qtcreator    
sudo apt-get install qt5-default 
echo "------------------------------------------------"
echo "----------- Installation Complete --------------"

```

## Install Docker and Nvidia-Docker script
Installation Script to use docker and nvidia-docker. [Reference Docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04) and [Reference Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker)

```
#!/bin/bash
# DOCKER
# Reference https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04
sudo apt update 
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
sudo apt install docker-ce
sudo usermod -aG docker ${USER}
su - ${USER}
docker info
docker run -it hello-world 
# be sure that $sudo systemctl status docker returns active. If this does not occur, do $sudo systemctl start docker
# if systemctl not found $sudo apt install systemd

# Optional NVIDIA-DOCKER
# if u are interested in using docker with nvidia u can dowload nvidia-docker
# Reference: https://github.com/NVIDIA/nvidia-docker
#curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
#distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
#curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
#sudo apt-get update
# Install nvidia-docker2 and reload the Docker daemon configuration
# sudo apt-get install -y nvidia-docker2
#sudo pkill -SIGHUP dockerd
# Test nvidia-smi with the latest official CUDA image
# docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```

## Installtion Script Let's Encrypt
We provide a script to create SSL certificate self-signed using Certbot. 

# Set up let's Encrypt

[Reference](https://certbot.eff.org/lets-encrypt/ubuntubionic-other)

```
apt-get update
apt-get install software-properties-common
add-apt-repository universe
add-apt-repository ppa:certbot/certbot
apt-get update
apt-get install certbot
```

Get Certificate:

```
certbot certonly --standalone
```

Insert domain name chosen: mydomain, Email:  myemail@gmail.com , then it will return the certificates!!
The email is important for the renowal! To update the email do:

```
sudo certbot update_account --email  myemail@gmail.com
```

Automatic renowal enable (we have enabled it):

```
certbot renew --dry-run
```

Certificate are contained available at:

```
    SSL_PRIVATE_KEY : '/etc/letsencrypt/live/mydomain/privkey.pem',
    SSL_CERTIFICATE : '/etc/letsencrypt/live/mydomain/fullchain.pem'
```

where mydomain is the domain name.


## Docker-Compose sample file

To use docker-compose it is required to have it installed [Reference](https://linuxize.com/post/how-to-install-and-use-docker-compose-on-ubuntu-18-04/).

Create a container made up by  a single image using docker-compose
```
version: '3.3'
services:

# docker-compose exec service_name bash
  service_name:
    image: image-name
    container_name: thermohuman
    volumes:
      - "$HOME/host_folder_name:/root/docker_folder_name"
```

More complex docker-compose file. The latter is a sample file to run tow node.js web_servers (web_server_1 and web_server_2) using nginx as reverse proxy. Furthermore we also have a postgres_database, a portainer gui and a vscode live server.
```
version: '3.3'
services:

# Nginx reverse proxy
  nginx-reverseproxy:
      image: nginx
      container_name: nginx_reverseproxy
      restart: always
      volumes:
        - "/root/docker/nginx.conf:/etc/nginx/conf.d/default.conf:ro"
        - "/etc/letsencrypt/live/domain_web_server_1/privkey.pem:/etc/letsencrypt/live/domain_web_server_1/privkey.pem"
        - "/etc/letsencrypt/live/domain_web_server_1/fullchain.pem:/etc/letsencrypt/live/domain_web_server_1/fullchain.pem"
        - "/etc/letsencrypt/live/domain_web_server_2/privkey.pem:/etc/letsencrypt/live/domain_web_server_2/privkey.pem"
        - "/etc/letsencrypt/live/domain_web_server_2/fullchain.pem:/etc/letsencrypt/live/domain_web_server_2/fullchain.pem"
      ports:
        - "80:80"
        - "443:443"
      command: [nginx-debug, '-g', 'daemon off;']


# Postgres database container setup
  postgres-database:
    image: postgres
    container_name: postgres_database
    restart: always
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres_password
    volumes:
        # Entry init.sql scrip to setup the database
      - "./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql" 
      - "db-data:/var/lib/postgresql/data"
    ports:
      - "5432:5432"

# Web Server 1
  web_server_1:
    image: "username/web_server_1"
    container_name: web_server_1
    depends_on:
      - postgres-database
    volumes:
     - "../web_server_1:/root/web_server_1"
     - "/etc/localtime:/etc/localtime:ro"
    restart: always
    ports:
     - "40001:80"
    command: bash -c "source /root/.bashrc && cd /root/web_server_1 && npm install &&  node index.js"

# Web Server 2
  web_server_2:
    image: "username/web_server_2"
    container_name: web_server_2
    depends_on:
      - postgres-database
    volumes:
     - "../web_server_2:/root/web_server_2"
     - "/etc/localtime:/etc/localtime:ro"
    restart: always
    #network_mode: "host"
    ports:
     - "40002:80"
    command: bash -c "source /root/.bashrc && cd /root/web_server_2 && npm install &&  node index.js"

# VSCODE Server
# docker run -it --name=myide_1 --net=host -v path_to/web_server_1:/home/coder/project codercom/code-server -p 30005  --no-auth
# docker run -it --name=myide_2 --net=host -v path_to/web_server_2:/home/coder/project codercom/code-server -p 30005  --no-auth
# with authentication --> docker run -it --net=host --name=myide  -v path_to/web_server_1:/home/coder/project codercom/code-server -p 30005
  codeserver:
      image: codercom/code-server
      container_name: vscode_server
      volumes:
        - "../web_server_1:/home/coder/project"
        # If u use https://github.com/jwilder/nginx-proxy (recommended)
        # environment:
        # VIRTUAL_HOST: your.domain.tld
        # VIRTUAL_PORT: 8443
      ports:
        # With SSL
        - "30005:30005"
        # With HTTP
        #- "80:8443"
      command: code-server -p 30005  --no-auth # --allow-http
      # Connect via network
      # networks:
      #   outpost_network:

# Portainer Server
# docker volume create portainer_data
# docker run -d -p 9000:9000 -p 8000:8000 --name portainer --restart always -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer
  portainer-server:
    image: portainer/portainer  
    container_name: portainer_server
    restart: always
    volumes:
      - "/var/run/docker.sock:/var/run/docker.sock"
      - "portainer_data:/data_portainer"
    ports:
      - "8000:8000"
      - "9000:9000"
      
# Postgres persistent volume, Portainer volume extra   
volumes:
  db-data:
  portainer_data:
```

nginx.conf example

```
# Useful examples
# https://www.digitalocean.com/community/questions/best-way-to-configure-nginx-ssl-force-http-to-redirect-to-https-force-www-to-non-www-on-serverpilot-free-plan-by-using-nginx-configuration-file-only


# Redirect http to https
server {
    listen 80;
    server_name domain_web_server_1 domain_web_server_2;
    return 301 https://$host$request_uri;
}

# Proxy pass to http webserver_1
server {
    listen      443  ssl;
    listen [::]:443  ssl;
    server_name domain_web_server_1;

    ssl on;
    ssl_certificate /etc/letsencrypt/live/domain_web_server_1/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/domain_web_server_1/privkey.pem;
    ssl_session_cache shared:SSL:10m;

    client_max_body_size 100M;

    location / {
        proxy_pass http://domain_web_server_1:40001;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;
    }
}

# Proxy pass to http webserver_2
server {
    listen      443 ssl;
    listen [::]:443 ssl;
    server_name  domain_web_server_2;

    ssl on;
    ssl_certificate /etc/letsencrypt/live/domain_web_server_2/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/domain_web_server_2/privkey.pem;
    ssl_session_cache shared:SSL:10m;

    client_max_body_size 100M;

    location / {
        proxy_pass http://domain_web_server_2:40002;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;
    }
}
```

## Screen Userful commands

# Screen instructions 

[Reference](https://linuxize.com/post/how-to-use-linux-screen/)

1. Install screen

```
sudo apt install screen
```

2. Create new screen

```
screen -S cmclima_webserver
screen -S cmclima_signalingserver
screen -S cmclima_turnserver
screen -S cmclima_servers
```

3. Attach to cmclima_webserver

```
screen -r cmclima_webserver
source $HOME/.bashrc && cd $HOME/webrtc-apprtc && node index.js
```

4. Attach to cmclima_signalingserver

```
screen -r cmclima_signalingserver
cd $HOME/signaling-go/bin/ && ./collidermain -port=30003 -tls=true -room-server="cmclima-vision.it:30003"
```

5. Attach to cmclima_turnserver screen

```
screen -r cmclima_turnserver
cd $HOME/webrtc-apprtc/scripts && turnserver -c  turnserver.conf
```

6. Detach from screen

```
CRTL+A and then D
```

7. To reattach

```
screen -r cmclima_server_screen # or
screen -d -r cmclima_server_screen
```

8. List active screen

```
screen -ls
```


## Run Multiple Bash Scripts inside the same terminal

```
#!/bin/bash

# First instruction
python script1.py & # &>/dev/null &

# Second Instruction
python script2.py & # &>/dev/null &

# Third Instruction
python script3.py 

```

## Open A terminal for each  script

```
#!/bin/bash

#---------------------------------------------------------------------------------------------
# INTERNAL PROCESSES
#---------------------------------------------------------------------------------------------
xfce4-terminal \
`#---------------------------------------------------------------------------------------------` \                                          ` \
`#---------------------------------------------------------------------------------------------` \
--tab --title "Terminal 1" --command "bash -c \"
python script1.py --wait;
exec bash\""  \
`#---------------------------------------------------------------------------------------------` \                                                                   ` \
`#---------------------------------------------------------------------------------------------` \
--tab --title "Terminal 2"	--command "bash -c \"
python script2.py  --wait;
exec bash\""  \
`#---------------------------------------------------------------------------------------------` \
`#---------------------------------------------------------------------------------------------` \
--tab --title "Terminal 3"	--command "bash -c \"
python script3.py  --wait;
exec bash\"" & 
```