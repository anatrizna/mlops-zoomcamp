FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN mkdir ./output

COPY [ "requirements.txt", "h4_answers.py", "./"]

RUN pip install -r requirements.txt
