FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY static/ static/
COPY knowledge/ knowledge/

EXPOSE 8901

CMD ["python3", "server.py"]
