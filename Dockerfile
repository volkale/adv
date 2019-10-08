FROM jupyter/scipy-notebook

WORKDIR /app

COPY . /app

USER root
RUN python setup.py install

EXPOSE 8765

VOLUME /app

CMD ["./bin/entrypoint.sh"]