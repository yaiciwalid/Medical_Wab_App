FROM python:3.9.16-slim

EXPOSE 8501

VOLUME app /app

WORKDIR app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip uninstall --yes protobuf
RUN pip install protobuf==4.21.12
RUN cp /usr/local/lib/python3.9/site-packages/google/protobuf/internal/builder.py /app/builder.py
RUN pip uninstall --yes protobuf
RUN pip install protobuf==3.19.6
RUN cat /app/builder.py
RUN mv  /app/builder.py /usr/local/lib/python3.9/site-packages/google/protobuf/internal/

ENTRYPOINT ["streamlit", "run", "./interface/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
