# Production-Ready Django Dockerfile

# --- Stage 1: The Builder ---
# This stage installs all dependencies, including build-time tools.
# Using a specific version tag is a best practice.
FROM python:3.10-slim-bullseye AS builder

# Set environment variables for Python to run optimally
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and define the path for the virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
# Add the venv's bin directory to the PATH for this stage
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install system dependencies that might be needed for Python packages.
# 'build-essential' is for compiling C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Set the working directory for the build
WORKDIR /app

# [OPTIMIZATION] Copy only the requirements files first to leverage Docker's layer cache.
COPY requirements/requirements.txt .
COPY requirements/requirements-ml.txt .

# Install the dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-ml.txt


# --- Stage 2: The Final Production Image ---
# This stage starts from a clean base image and copies only what's necessary,
# resulting in a smaller and more secure final image.
FROM python:3.10-slim-bullseye AS final

# Create a non-root user and group for security
RUN groupadd -r django && useradd -r -g django django

# Copy the entire virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the working directory for the application
WORKDIR /home/django/app

# Copy the application source code from your local `src/web` directory
COPY src/web/ .

# Copy the ML models into the correct application sub-directory
COPY src/ml/models ./webbidder/

# Change ownership of the app directory and its contents to the non-root user
RUN chown -R django:django /home/django/app

# Switch to the non-root user. Subsequent commands will run as 'django'.
USER django

# Expose the port that Gunicorn will listen on
EXPOSE 8000

# Run the application with Gunicorn
# We call the gunicorn executable directly from our virtual environment's bin directory.
# This ensures it uses the correct Python interpreter and can find all installed packages.
CMD ["/opt/venv/bin/gunicorn", "--bind", "0.0.0.0:8000", "powerbidder.wsgi:application"]