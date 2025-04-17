# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.9-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME="/home/user"
WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . ./

# Expose the port app will run on
EXPOSE 7860

# Start the app
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:server"] 