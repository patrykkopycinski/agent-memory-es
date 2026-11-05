FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the local embedding model (AMES_EMBED_BACKEND=local) so the container never needs the
# network at runtime: ~470 MB ONNX weights + 17 MB tokenizer, pinned to one HF revision.
# Keep E5_REV in step with LOCAL_MODEL_REVISION in app/embeddings.py.
ARG E5_REPO=intfloat/multilingual-e5-small
ARG E5_REV=614241f622f53c4eeff9890bdc4f31cfecc418b3
ENV E5_REPO=$E5_REPO E5_REV=$E5_REV AMES_LOCAL_MODEL_DIR=/opt/models/multilingual-e5-small
RUN python -c "import os, urllib.request as u; d=os.environ['AMES_LOCAL_MODEL_DIR']; r=os.environ['E5_REPO']; v=os.environ['E5_REV']; [(os.makedirs(os.path.dirname(os.path.join(d,f)), exist_ok=True), u.urlretrieve(f'https://huggingface.co/{r}/resolve/{v}/{f}', os.path.join(d,f))) for f in ('onnx/model.onnx','tokenizer.json')]"
COPY app ./app
EXPOSE 8123
