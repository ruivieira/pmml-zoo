FROM registry.access.redhat.com/ubi8/ubi-minimal:8.1
WORKDIR /work/
COPY pyproject.toml /work/
COPY src/* /work/
COPY src/openapi/* /work/openapi/

RUN microdnf install -y python3 \
    && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 - \
    && ~/.poetry/bin/poetry install

EXPOSE 5000

CMD ["/root/.poetry/bin/poetry", "run", "python", "pmmlzoo.py"]