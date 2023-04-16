FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Get the data, we'll use it to train the SVC and KNN models
RUN wget https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
RUN gzip -d covtype.data.gz

# No need to use init_api.py, since currently the model is so small it can be stored in the github repo
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]