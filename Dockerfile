FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY gpumem ./gpumem
RUN pip install --no-cache-dir . && useradd --uid 10001 --no-create-home gpumem
USER 10001
# On GPU nodes the NVIDIA container toolkit injects libnvidia-ml when these are set.
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=utility
ENTRYPOINT ["gpumem"]
CMD ["watch", "--interval", "10"]
