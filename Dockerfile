FROM jupyter/scipy-notebook:814ef10d64fb

WORKDIR /app

COPY . /app

USER root
RUN conda install gcc_linux-64 gxx_linux-64 -c anaconda
RUN python setup.py install
RUN pip install -r requirements.txt

EXPOSE 8765

VOLUME /app

CMD ["./bin/entrypoint.sh"]