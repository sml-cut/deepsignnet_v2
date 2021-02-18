#!/bin/sh

apt install -y libgl1-mesa-glx

python -m django startproject config .
python manage.py startapp slt
python manage.py migrate
#jupyter notebook --ip 0.0.0.0 --port 8889 --no-browser --allow-root

exec "$@"
