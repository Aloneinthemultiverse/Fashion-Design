# Hugging Face Space (Docker SDK, free CPU tier). Built by scripts/deploy_space.py,
# which stages the code plus a compacted copy of the indexed outfit images.
FROM python:3.12-slim
RUN pip install --no-cache-dir uv && useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user pyproject.toml uv.lock README.md ./
COPY --chown=user src ./src
# CPU-only torch: the CUDA wheels are gigabytes the free tier cannot use.
RUN uv pip install --system --no-cache --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      ".[embed,vision,ui]" python-dotenv
COPY --chown=user data ./data
USER user
# Keys come from the Space's secrets: FASHION_GEMINI_API_KEY, FASHION_POLLINATIONS_TOKEN.
ENV FASHION_VISION_PROVIDER=gemini \
    FASHION_EMBED_PROVIDER=openclip \
    FASHION_STORE_PROVIDER=memory \
    FASHION_GENERATION_PROVIDER=pollinations \
    FASHION_TRYON_PROVIDER=pollinations \
    HF_HOME=/home/user/.cache/huggingface
EXPOSE 7860
CMD ["streamlit", "run", "src/fashion/ui/app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.headless", "true"]
