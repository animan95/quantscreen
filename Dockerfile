FROM ubuntu:22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Basic setup
RUN apt-get update && apt-get install -y \
    wget curl git build-essential cmake libglib2.0-0 \
    libxrender1 libsm6 libxext6 libboost-all-dev python3 python3-pip python3-dev \
    python3-venv libeigen3-dev libomp-dev libopenblas-dev \
    && apt-get clean

# Set up Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install RDKit
RUN pip install rdkit-pypi

# Install PySCF and other packages
RUN pip install pyscf streamlit py3Dmol pandas numpy seaborn matplotlib joblib

# Copy your app
WORKDIR /app
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

