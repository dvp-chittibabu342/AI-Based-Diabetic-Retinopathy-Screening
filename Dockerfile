FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 7860
ENV PORT=7860
CMD ["python", "webapp/run.py"]