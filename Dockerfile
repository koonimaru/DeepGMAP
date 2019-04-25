FROM tensorflow/tensorflow:1.9.0-rc2-gpu-py3
RUN apt-get update && apt-get install -y --no-install-recommends bedtools git
RUN pip3 install --no-cache-dir "setuptools<=39.1.0" matplotlib
RUN git clone https://github.com/koonimaru/DeepGMAP.git && \
	cd DeepGMAP && git checkout && \
	python3 setup.py install