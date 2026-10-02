FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Mount .env and data/ at runtime; never bake keys into the image.
CMD ["python", "-m", "jevbot", "run"]
