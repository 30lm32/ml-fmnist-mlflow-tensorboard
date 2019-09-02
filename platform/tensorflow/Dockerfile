FROM tensorflow/tensorflow:latest-py3

RUN pip install --upgrade pip

COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN rm ./requirements.txt
