# Step 1: Start from an official Python 3.10 image
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file and install dependencies
# This now correctly finds requirements.txt in the root context
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your backend application code into the container
# THIS IS THE KEY CHANGE:
COPY backend/ .

# Step 6: Expose the port the app runs on
EXPOSE 8000

# Step 7: Define the command to run your application
# We reference main.py which is now in the /app directory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]