FROM continuumio/miniconda3:4.9.2

RUN apt-get --allow-releaseinfo-change-suite update && apt-get install -y \
        unzip \
        git \
        python3-pip \
        python3-numpy \
        python3-scipy \
        python3-pandas \
    && apt clean

WORKDIR /opt/conda
RUN /opt/conda/bin/conda install jupyter -y --quiet \
    && /opt/conda/bin/conda install pydotplus -y --quiet \
    && /opt/conda/bin/conda install tensorflow -y --quiet \ 
    && mkdir /opt/notebooks 

RUN pip install -U pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /opt/notebooks

RUN git clone https://github.com/luisguiserrano/manning.git
RUN cp -R manning/Chapter_* .
RUN rm -rf manning
CMD ["/opt/conda/bin/jupyter", "notebook", "--notebook-dir=/opt/notebooks", "--ip=*", "--port=8888", "--no-browser", "--allow-root"]
