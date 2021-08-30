FROM python:3.8

WORKDIR /tmp
COPY ./requirements.txt /tmp
RUN pip install -r requirements.txt \
    && rm /tmp/requirements.txt

ARG CACHE_DATE

RUN mkdir -p /home/ampas/idt-calculator
WORKDIR /home/ampas/idt-calculator
COPY . /home/ampas/idt-calculator

ENV AMPAS_APPS_CSS=/home/ampas/idt-calculator/assets/style.css

CMD sh -c 'if [ -z "${SSL_CERTIFICATE}" ]; then \
    gunicorn --log-level debug -b 0.0.0.0:8000 index:SERVER; else \
    gunicorn --certfile "${SSL_CERTIFICATE}" --keyfile "${SSL_KEY}" --log-level debug -b 0.0.0.0:8000 index:SERVER; fi'
