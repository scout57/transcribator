FROM python:3.12-slim

RUN apt-get update
RUN apt-get install -y ffmpeg --no-install-recommends
RUN rm -rf /var/lib/apt/lists/*

# COPY --from=mwader/static-ffmpeg:4.4.1 /ffmpeg /usr/local/bin/
# COPY --from=mwader/static-ffmpeg:4.4.1 /ffprobe /usr/local/bin/
# COPY --from=mwader/static-ffmpeg:4.4.1 /qt-faststart /usr/local/bin/
