python -m vllm.entrypoints.openai.api_server \
    --model casperhansen/llama-3-70b-instruct-awq \
    --dtype=half \
    --tensor-parallel-size=1 \
    --quantization awq