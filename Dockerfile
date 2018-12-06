FROM ubuntu:16.04

#Install the tools needed to run the code
RUN apt-get update -qq
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -qy python3 python3-pip python3-dev
RUN apt-get install -qy build-essential g++

# Install python libraries
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY src /app
WORKDIR /app
ENTRYPOINT python3 train_prospection.py ${base_vehicular} ${data_renovaciones} ${parque_vehicular}