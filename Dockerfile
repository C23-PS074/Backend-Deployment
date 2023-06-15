# Base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the code and requirements files to the working directory
COPY main.py .
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Install Protocol Buffers compiler
RUN apt-get update && \
    apt-get install -y protobuf-compiler

# Clone the TensorFlow models repository if it doesn't exist
RUN if [ ! -d "models" ]; then git clone --depth 1 https://github.com/tensorflow/models; fi

# Change to the models/research directory
WORKDIR /app/models/research

# Install the TensorFlow Object Detection API
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py .
RUN python -m pip install .

# Change back to the app directory
WORKDIR /app

# Expose the necessary port (if applicable)
# EXPOSE 8080

# Run the application
CMD [ "python", "main.py" ]