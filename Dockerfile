# 1. Base image with Python
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# 2. Set working directory inside the container
WORKDIR /workspace

# 3. Copy your files into the container
COPY requirements.txt .
COPY src /workspace/src
COPY model /workspace/model
COPY images /workspace/images

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt 

# 5. make the script executable inside the Docker image
RUN chmod +x "src/RootXplorer_pipeline.sh"

# 6. add the entrypoint
ENTRYPOINT ["bash", "workspace/src/RootXplorer_pipeline.sh"]
# CMD [ "bash" ]