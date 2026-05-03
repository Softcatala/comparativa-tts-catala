FROM ghcr.io/softcatala/comparativa-tts-catala-base:latest

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

# models/ is populated by the GitHub Actions workflow from the HF space LFS
COPY --chown=user models models

COPY --chown=user engine.py .
COPY --chown=user mms.py .
COPY --chown=user festival.py .
COPY --chown=user app.py .

RUN mkdir -p cache && chmod 777 cache

ENV NUMBA_CACHE_DIR=/home/user/cache
ENV MPLCONFIGDIR=/home/user/cache

EXPOSE 7860

CMD ["python", "app.py"]
