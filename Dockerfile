# Use an appropriate Python base image
FROM python:3.9-slim-buster

# Set environment variables to avoid buffer and bytecode issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install required libraries for NumPy and other dependencies
RUN apt-get update && apt-get install -y \
    libblas-dev \
    liblapack-dev \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port the Flask app runs on
EXPOSE 5001

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

# Run the Flask application
CMD ["flask", "run"]
