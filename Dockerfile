FROM python:3.11
WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt
EXPOSE 8501
CMD [ "streamlit","run","california.py" ]
