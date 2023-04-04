FROM python:3.8

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "attendance_system.py"]
