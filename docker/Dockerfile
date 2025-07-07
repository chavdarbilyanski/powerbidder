# Production-Ready Django Dockerfile

# Use a specific, non-root user for security
FROM python:3.10-slim AS base

# --- Stage 1: Build Dependencies ---
# This stage installs dependencies, which don't change often,
# allowing Docker to cache this layer effectively.
FROM base AS builder
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Create a virtual environment inside the container
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install system dependencies that might be needed for Python packages
# (e.g., postgresql-client for psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements/requirements.txt .
COPY requirements/requirements-ml.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-ml.txt



# --- Stage 2: Final Production Image ---
# This stage copies only what's necessary, resulting in a smaller, more secure image.
FROM base
# Create a non-root user and group
RUN groupadd -r django && useradd -r -g django django

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the path to use the venv
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /home/django/app

# Copy the application source code
COPY src/web/ .

# Copy the training model
COPY src/ml/models ./webbidder/

# Change ownership of the files to the non-root user
RUN chown -R django:django /home/django/app

# Switch to the non-root user
USER django

# Expose the port
EXPOSE 8000

# Run Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "powerbidder.wsgi:application"]
# --- DEBUGGING CMD ---
# This command just sleeps forever, keeping the container alive for us to inspect.
# CMD ["tail", "-f", "/dev/null"]