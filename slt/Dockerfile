# pull official base image
FROM python:3.7.9

# set work directory
WORKDIR /usr/src/slt

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install psycopg2 dependencies
RUN apt-get update # && apk add postgresql-dev gcc python3-dev musl-dev

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# copy entrypoint.sh
COPY ./entrypoint.sh .

# copy project
COPY . .

# run entrypoint.sh
ENTRYPOINT ["/usr/src/slt/entrypoint.sh"]

# CMD ["jupyter", "notebook", "--port=8889", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
