# Build context: backend/  ->  docker build -f mcp_server/Dockerfile -t ames-mcp backend/
FROM python:3.12-slim
WORKDIR /srv
COPY mcp_server/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir -r /tmp/req.txt
COPY mcp_server /srv/mcp_server
ENV PYTHONUNBUFFERED=1 AMES_URL=http://host.docker.internal:8123
ENTRYPOINT ["python", "-m", "mcp_server"]
