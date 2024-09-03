#pull python base image

FROM python:3.12

ADD heartdisease_model_api heartdisease_model_api 

WORKDIR /heartdisease_model

RUN pip install --upgrade pip

RUN pip install -r requirements.txt 

EXPOSE 8001

CMD ["python", "app/main.py"]

