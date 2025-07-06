# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements files
COPY requirements.txt .
COPY requirements-dev.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Django project code
COPY src/web/ .

# Set environment variables
# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures Python output is unbuffered for logging
ENV PYTHONUNBUFFERED=1

# Expose port 8000 for the Django app
EXPOSE 8000

# Run Gunicorn to serve the Django app
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "myproject.wsgi:application"]