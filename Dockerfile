# pull official base image
FROM python:latest

USER root

RUN pip install --upgrade pip

# install required python packages
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install requests

# copy project
RUN mkdir -p /usr/src/app/experiment
COPY ./experiment/ /usr/src/app/experiment

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app/experiment"

# set work directory
WORKDIR /usr/src/app/experiment

ENTRYPOINT python runner.py