FROM python:3.7.0

RUN pip install --upgrade pip

COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN rm ./requirements.txt

EXPOSE 5000

RUN mkdir -p /server
WORKDIR /server
COPY ./wait-for-it.sh ./
RUN chmod +x wait-for-it.sh

RUN mkdir /mlflow/

CMD ./wait-for-it.sh postgres:5432 -- mlflow server \
    --backend-store-uri postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@postgres:5432/mlflow \
    --default-artifact-root /mlflow/ \
    --host 0.0.0.0