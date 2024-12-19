# Base Python image
    FROM python:3.10-slim

    WORKDIR /app

    COPY mock.py /app/mock.py

    RUN pip install --no-cache-dir flask requests

    CMD ["python", "/app/mock.py"]
    