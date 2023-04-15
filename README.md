# Internship Task For Openx

This repository contains code for my internship task for OpenX. 

## API Documentation

### How to run the Docker image

1. Clone the repository
2. Run `docker build -t app .` to build the image
3. Run `docker run -p 8000:8000 app` to run the image and map the port 8000 

At that stage, you should be able to access the API at `0.0.0.0:8000`.

### How to access the API

You can access the auto-generated API documentation at `0.0.0.0:8000/docs`.

Should this not be enough, here's a quick overview of the API:

* There are 4 endpoints: 
    1. `/api/heuristic`, being a very simple heuristic model,
    2. `/api/svc`, being a support vector classifier,
    3. `/api/knn`, being a k-nearest neighbors classifier, and
    4. `/api/neural_net` being a neural network classifier.
* Each endpoint takes a GET request with only one parameter: `data`, being a 54-element int array, representing features of the covtype dataset.
* Each endpoint returns a JSON object with a single key, `result`, which is the predicted class from the covtype dataset.