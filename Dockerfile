FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Get the data, we'll use it to train the SVC and KNN models
RUN wget https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
RUN gzip --force -d covtype.data.gz

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]