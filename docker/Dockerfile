ARG base_image=gcr.io/transyto-image
FROM ${base_image}

ENV PATH="/root/miniconda3/bin:${PATH}"
ENV TRANSYTO=/home/transyto

USER root
# update apt-get
RUN apt-get update -y && \
    apt-get install -y wget ca-certificates git python3-pip vim && \
    # Install conda
    wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    conda init bash && \
    . ~/.bashrc && \
    # Create the conda env
    conda install numpy astropy ipython pymongo matplotlib pyyaml pandas && \
    # Install extra packages
    pip install astroquery panoptes-utils && \
    # Downgrade coverage version to avoid pytest issues
    pip uninstall -y coverage && \
    pip install coverage==4.5.4 && \
    # Make huntsman directories
    mkdir -p "${TRANSYTO}" && \
    cd /home

# Make sure conda env is used by default for remaining RUN commands
SHELL ["conda", "run", "/bin/bash", "-c"]

# Install local transyto into the image
COPY . ${TRANSYTO}
RUN cd ${TRANSYTO} && python setup.py develop

WORKDIR /home
CMD ["/bin/bash"]
