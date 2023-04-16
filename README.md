# Internship Task For Openx

This repository contains code for my internship task for OpenX. 

## API Documentation

### How to run the Docker image

1. Clone the repository
2. Run `docker build -t app .` to build the image
3. Run `docker run -p 8000:8000 app` to run the image 

At that stage, you should be able to access the API at `0.0.0.0:8000`.

### How to access the API

You can access the auto-generated API documentation at `0.0.0.0:8000/docs`.

Should this not be enough, here's a quick overview of the API:

* There are 4 endpoints, one for each model (similiarly to how HuggingFace does it with `InferenceAPI`): 
    1. `/api/heuristic`, being a very simple heuristic model,
    2. `/api/svc`, being a support vector classifier,
    3. `/api/knn`, being a k-nearest neighbors classifier, and
    4. `/api/neural_net` being a neural network classifier.
* Each endpoint takes a POST request with only one parameter: `data`, being a 54-element int array, representing features of the covtype dataset.
* While these are POST requests, they are not mutating the state of the server in any way. I've chosen the POST method, because we somehow need to pass the data to the server, and GET requests are not supposed to do that.
* Each endpoint returns a JSON object with a single key, `result`, whose value is the predicted class from the covtype dataset.