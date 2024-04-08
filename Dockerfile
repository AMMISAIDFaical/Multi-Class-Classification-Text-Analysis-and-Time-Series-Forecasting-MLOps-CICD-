FROM python:latest
LABEL authors="Faical"
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    bash git openssh-client

WORKDIR ./app

COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 8000
COPY ./api /app
COPY ./common.py .
COPY ./config.ini .

#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["python", "main.py"]
