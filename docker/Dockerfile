ARG base_image=gcr.io/panoptes-exp/panoptes-plate-solver
FROM ${base_image}

ENV PATH="/root/miniconda3/bin:${PATH}"
ENV HUNTSMAN_LOG_DIR=/home/huntsman-logs
ENV HUNTSMAN_DRP=/home/huntsman-drp

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
    mkdir -p "${HUNTSMAN_LOG_DIR}" && \
    # Install astro_metadata_translator from source && \
    cd /home && \
    git clone https://github.com/lsst/astro_metadata_translator.git && \
    cd astro_metadata_translator && \
    pip install -e .

# Make sure conda env is used by default for remaining RUN commands
SHELL ["conda", "run", "/bin/bash", "-c"]

# Install local huntsman-drp into the image
COPY . ${HUNTSMAN_DRP}
RUN cd ${HUNTSMAN_DRP} && python setup.py develop

WORKDIR /home
CMD ["/bin/bash"]
