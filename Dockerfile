FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY app ./app
COPY configs ./configs
COPY docs ./docs
COPY release ./release
COPY data/manual ./data/manual
COPY .streamlit ./.streamlit

ENV PORT=8501
EXPOSE 8501

CMD ["sh", "-c", "streamlit run app/Home.py --server.address 0.0.0.0 --server.port ${PORT}"]
