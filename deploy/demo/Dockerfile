# syntax=docker/dockerfile:1
# repository root:
#   docker build -f deploy/demo/Dockerfile -t spatialaim-demo \
#     --build-arg SPATIALAIM_VERSION=1.4.0 .
FROM python:3.13-slim

# setuptools-scm needs the version without the .git folder in the build context.
ARG SPATIALAIM_VERSION=0.0.0+demo
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SPATIAL_AIM=${SPATIALAIM_VERSION} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/spatialaim
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . && rm -rf src

# Plotly.js for the client-side plot components, served by Streamlit itself so
# visitors' browsers contact no CDN; the same version the components otherwise
# import from esm.sh.
ARG PLOTLY_JS_VERSION=2.35.2
RUN python - <<EOF
import pathlib, urllib.request
import spatialaim.gui
static = pathlib.Path(spatialaim.gui.__file__).parent / "static"
static.mkdir(exist_ok=True)
urllib.request.urlretrieve(
    "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@${PLOTLY_JS_VERSION}/plotly.min.js",
    static / "plotly.min.js",
)
EOF

COPY deploy/demo/legal ./legal

RUN useradd --create-home --uid 10001 spatialaim
USER spatialaim
# numba (via scanpy/umap) caches compiled code next to the sources by default.
ENV NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["spatialaim", "gui", "--demo_data", "/data", "--legal_dir", "/opt/spatialaim/legal", "--server_port", "8501"]
