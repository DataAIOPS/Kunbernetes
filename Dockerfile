# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY house_price_model.joblib requirements.txt app.py /app

# Ensure the Image Includes Necessary Shells
RUN apt-get update && apt-get install -y bash

# Install Flask and scikit-learn
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
