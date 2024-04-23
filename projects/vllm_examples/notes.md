# Resources

- Usage stats collection: <https://docs.vllm.ai/en/latest/serving/usage_stats.html>


# Phoenix Cluster

## RTX-6000

### Llama-3 8B

- fp16
  - 34 tkps
  - bs=1
  - 2 minutes to generate 4096 tokens
  - Can fit full conditional context
  - generated full 750 token response in 22 seconds

### LLama-3 70B

Cannot fit without tensor-parallelism
Tensor parallelism is not enough
Can fit with awq quantization

- fp16, tensor parallel, awq quantization
    - 2 tkps
    - bs=1
    - 1 minute to read 1500 tokens and generate 700 tokens
