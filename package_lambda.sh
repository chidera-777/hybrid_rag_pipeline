#!/bin/bash

# Create package directory
mkdir -p lambda_package
cd lambda_package

# Copy your code
cp -r ../mlops .
cp -r ../ingestion .
cp -r ../vectorstore .
cp -r ../schemas .
cp ../config.py .
cp ../lambda/build_index.py handler.py

# Install dependencies
pip install -r ../requirements.txt -t .

# Create zip
zip -r ../lambda_function.zip .

cd ..
rm -rf lambda_package

echo "Lambda package created: lambda_function.zip"