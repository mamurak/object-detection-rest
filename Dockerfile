FROM registry.access.redhat.com/ubi8/python-38

USER root

RUN dnf -y update \
 && dnf -y install python3-pip \
 && dnf -y clean all \
 && rm -rf /var/cache/dnf \
 && pip install pip==22.3.1 setuptools==65.3.0

USER 1001

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 uninstall -y opencv-python && \
    pip3 uninstall -y opencv-python-headless && \
    pip3 install opencv-python-headless==4.8.0.74

COPY *.py ./

CMD FLASK_APP=wsgi.py flask run
