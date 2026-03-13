FROM ubuntu:22.04
MAINTAINER Mingxun Wang "mwang87@gmail.com"

RUN apt-get update && apt-get install -y build-essential libarchive-dev wget vim git-core

# Install Mamba
ENV CONDA_DIR /opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
# RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Adding to bashrc
RUN echo "export PATH=$CONDA_DIR:$PATH" >> ~/.bashrc

# Forcing version of Python
RUN mamba create -n python310 python=3.10 -y
RUN /bin/bash -c 'source activate python310 && python -m pip install --no-cache-dir --upgrade pip uv'

COPY requirements-streamlit.txt .
RUN /bin/bash -c 'source activate python310 && uv pip install --python $(which python) --no-cache -r requirements-streamlit.txt'

# google-genai (new SDK)
RUN /bin/bash -c 'source activate python310 && uv pip install --python $(which python) --no-cache python-dotenv'

# mcp
RUN /bin/bash -c 'source activate python310 && uv pip install --python $(which python) --no-cache mcp pydantic_ai'

COPY . /app
WORKDIR /app
