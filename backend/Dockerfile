FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the local embedding model (AMES_EMBED_BACKEND=local) so the container never needs the
# network at runtime: ~470 MB ONNX weights + 17 MB tokenizer, pinned to one HF revision and
# verified by sha256 (a mismatch fails the build). The build writes /opt/models/.../REVISION;
# app/embeddings.py refuses a model dir whose marker differs from LOCAL_MODEL_REVISION, and
# tests/test_local_model_loading.py fails if E5_REV drifts from that constant.
ARG E5_REPO=intfloat/multilingual-e5-small
ARG E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3
ARG E5_MODEL_SHA256=ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665
ARG E5_TOKENIZER_SHA256=0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39
ENV AMES_LOCAL_MODEL_DIR=/opt/models/multilingual-e5-small
COPY scripts/bake_e5_model.py /tmp/bake_e5_model.py
RUN python /tmp/bake_e5_model.py --dest "$AMES_LOCAL_MODEL_DIR" --repo "$E5_REPO" --rev "$E5_REV" \
      --sha "onnx/model.onnx=$E5_MODEL_SHA256" --sha "tokenizer.json=$E5_TOKENIZER_SHA256" \
    && rm /tmp/bake_e5_model.py
COPY app ./app
EXPOSE 8123
