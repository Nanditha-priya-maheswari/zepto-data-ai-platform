# Use a lightweight Python version
FROM python:3.10-slim

# Set the working folder inside the container
WORKDIR /app

# Copy all your project files into the container
COPY . /app

# Install the required libraries
RUN pip install --no-cache-dir -r requirements.txt

# Tell the container what to run when it starts
CMD ["python", "genai/assistant.py"]