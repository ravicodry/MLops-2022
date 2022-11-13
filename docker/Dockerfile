#FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./api/* /exp/api/
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD ["python", "./plot_graphs.py"]
WORKDIR /exp/api
CMD ["python","./app.py"]
EXPOSE 5000
