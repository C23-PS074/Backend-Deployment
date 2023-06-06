# Base image
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables if needed
# ENV

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "main.py"]
