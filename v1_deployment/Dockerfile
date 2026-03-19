# Use a slim Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:${PATH}"

# Install system dependencies (curl for Ollama installation)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Add a non-root user expected by Hugging Face Spaces (user ID 1000)
RUN useradd -m -u 1000 user

# Install Ollama (installed globally in /usr/local/bin)
# Note: installation as root is required for global binary placement
RUN curl -fsSL https://ollama.com/install.sh | sh

# Switch to the non-root user
USER user

# Set the working directory
WORKDIR /home/user/app

# Copy the Python requirements and install them securely
COPY --chown=user:user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Start the Ollama server in the background and pull the model
# Using `user` home directory means weights are stored in /home/user/.ollama
RUN nohup bash -c "ollama serve &" && \
    sleep 5 && \
    ollama pull llama3.2-vision

# Copy the application code and data directories
# Make sure "index/" and "extracted_data/" are available in the build context
COPY --chown=user:user app.py .
COPY --chown=user:user index/ index/
COPY --chown=user:user extracted_data/ extracted_data/

# Copy the entrypoint script and make it executable
COPY --chown=user:user entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose the port Streamlit runs on (default HF Spaces port is 7860)
EXPOSE 7860

# Set the entrypoint to our custom script
CMD ["./entrypoint.sh"]
