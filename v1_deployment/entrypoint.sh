#!/bin/bash
# Start Ollama service in the background
ollama serve &

# Wait a moment for Ollama to be ready
sleep 5

# Start the Streamlit application on the port expected by HF Spaces
exec streamlit run app.py --server.address 0.0.0.0 --server.port 7860
