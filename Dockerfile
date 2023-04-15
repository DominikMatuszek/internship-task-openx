FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./

#RUN conda config --append channels conda-forge
#RUN conda create --name app --file requirements.txt 
#RUN conda activate app

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No need to use init_api.py, since currently the model is so small it can be stored in the github repo
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]