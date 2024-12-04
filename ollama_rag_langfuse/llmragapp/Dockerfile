FROM python:latest

WORKDIR /app

COPY requirements.txt ./
COPY app.py ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8051
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8051", "--server.address=0.0.0.0"]
