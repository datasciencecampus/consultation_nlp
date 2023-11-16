FROM python:3.9 

RUN git clone -b codespace_test https://github.com/datasciencecampus/consultation_nlp
WORKDIR "./consultation_nlp"

RUN pip install -r requirements.txt 

CMD ["streamlit", "run", "streamlit_app.py"]
