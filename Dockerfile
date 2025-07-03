FROM python:3.12.9-slim-bookworm

# Avoid interactive prompts (e.g., tzdata)
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages required for RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    libboost-all-dev \
    cmake \
    git \
    libeigen3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*


# Set working directory inside container
WORKDIR /GSK3BPred

RUN python3.12 -m venv /GSK3BPred/gsk3bpred
COPY dependencies.txt .

RUN . /GSK3BPred/gsk3bpred/bin/activate && pip install --upgrade pip && pip install -r dependencies.txt

RUN rm /GSK3BPred/dependencies.txt

# Clean up unnecessary build dependencies to reduce size
RUN apt-get purge -y build-essential python3-dev cmake libboost-all-dev libeigen3-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache


# Copy prediction script and models
COPY mordred_dnn_model.h5 mordred_scaler.pkl prediction_script_gsk3bpred.py X_train.csv ./

WORKDIR /WorkPlace
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
# Entry point: run your prediction script
ENTRYPOINT ["/GSK3BPred/gsk3bpred/bin/python", "/GSK3BPred/prediction_script_gsk3bpred.py"]

