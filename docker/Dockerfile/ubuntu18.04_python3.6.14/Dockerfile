From ubuntu:18.04

RUN apt-get -y update
RUN apt-get install -y wget
RUN apt-get -y install zip
RUN apt-get -y install vim
RUN apt-get -y install libbz2-dev
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-dev
RUN apt install build-essential -y
RUN apt install libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev -y
RUN apt install zlib1g-dev -y
RUN apt install wget -y
RUN apt install openssl -y
RUN apt install curl -y
RUN apt install libsqlite3-dev -y
RUN apt-get install -y liblzma-dev

# Install Python
WORKDIR /root
RUN wget https://www.python.org/ftp/python/3.6.14/Python-3.6.14.tgz
RUN tar -xvf Python-3.6.14.tgz
WORKDIR  /root/Python-3.6.14
RUN ./configure --enable-loadable-sqlite-extensions
RUN make && make install
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/local/bin/pip3 /usr/bin/pip
RUN ln -s /usr/local/bin/python3 /usr/bin/python

WORKDIR /home