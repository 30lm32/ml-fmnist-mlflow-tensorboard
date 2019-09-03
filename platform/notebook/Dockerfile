FROM jupyter/minimal-notebook

WORKDIR work

USER root
COPY ./package.list ./
RUN apt-get update
RUN cat package.list | grep -vE "^\s*#" | xargs apt-get install
RUN rm ./package.list

USER jovyan:100
RUN pip install --upgrade pip
RUN pip uninstall -y numpy

COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN rm ./requirements.txt

# Please, uncomment to disable token
#CMD ["jupyter", "notebook", "--no-browser","--NotebookApp.token=''","--NotebookApp.password=''"]


