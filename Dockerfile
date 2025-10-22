FROM python:3.10

COPY requirements.txt .
RUN pip3 install --timeout 1000 -r requirements.txt

RUN mkdir /app
WORKDIR /app

COPY proto/inference.proto .
RUN ls -la
COPY http-server.py .
COPY grpc-server.py .
COPY run_codegen.py .

RUN python3 run_codegen.py
COPY supervisord.conf .

EXPOSE 8080
EXPOSE 9090

ENTRYPOINT ["supervisord", "-c", "supervisord.conf"]