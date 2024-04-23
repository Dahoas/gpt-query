python -m vllm.entrypoints.openai.api_server \
    --model TechxGenus/Meta-Llama-3-70B-AWQ \
    --dtype=half \
    --tensor-parallel-size=4 \
    --quantization awq