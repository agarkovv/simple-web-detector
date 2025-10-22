FROM python:3.10

COPY requirements.txt .
RUN pip3 install --timeout 1000 -r requirements.txt

RUN mkdir /app
WORKDIR /app

RUN ls -la
COPY http-server.py .
COPY static ./static

EXPOSE 8080

ENTRYPOINT ["python3", "/app/http-server.py"]