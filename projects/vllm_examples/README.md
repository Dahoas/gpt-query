# Installation Notes

## PACE

Step 1: pip install vllm
```bash
python -m venv vllm
source vllm/bin/activate
pip install vllm
```
Step 2: Fix incompatible urllib3 version
```bash
pip uninstall urllib3
pip install urllib3==1.21.1
```

# Serve

```bash
python -m vllm.entrypoints.openai.api_server --model gpt2 \
--dtype half
```

List of models: (https://docs.vllm.ai/en/latest/models/supported_models.html#supported-models)[here]