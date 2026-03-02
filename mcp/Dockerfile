FROM ubuntu:22.04
MAINTAINER Mingxun Wang "mwang87@gmail.com"

RUN apt-get update && apt-get install -y build-essential libarchive-dev wget vim git-core

# Install Mamba
ENV CONDA_DIR /opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Adding to bashrc
RUN echo "export PATH=$CONDA_DIR:$PATH" >> ~/.bashrc

# Forcing version of Python
RUN mamba create -n python310 python=3.10 -y

COPY requirements.txt .
RUN /bin/bash -c 'source activate python310 && pip install -r requirements.txt'

# rapidfuzz
RUN /bin/bash -c 'source activate python310 && pip install requests fastmcp rapidfuzz'

COPY . /app
WORKDIR /app

# set the entry point for mcp server
# source activate python310 && python /app/growth_server.py
ENTRYPOINT ["/bin/bash", "-c", "source activate python310 && python /app/wom_mcp_server.py"]