FROM registry.access.redhat.com/ubi8/python-38

USER root

RUN dnf -y update \
 && dnf -y install python3-pip \
 && dnf -y clean all \
 && rm -rf /var/cache/dnf \
 && pip install pip==22.3.1 setuptools==65.3.0

USER 1001

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY *.py ./

CMD python3 /opt/app-root/bin/gunicorn wsgi --config gunicorn_config.py
