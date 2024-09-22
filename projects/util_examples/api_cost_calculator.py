"""
Script to quickly calculate costs for an experiment with an estimated average input tokens `input_len`, 
average output tokens `output_len` and `num_queries` queries. Cost is estimated in USD.
Note: we make the following assumptions
- Each query can generate the request number of output tokens in full (without the need for iterative prompting)
- A token is roughly the same between different providers
- Only computing text input/output
- Pricing checked on: 3/20/2024
    - OAI: https://openai.com/pricing
    - Anthropic: https://www.anthropic.com/api
    - Google: https://cloud.google.com/vertex-ai/generative-ai/pricing
"""
import argparse

THOUSAND = 1_000
MILLION = 1_000_000

model_tokens =  {
    "gpt-3.5-turbo": {"input_token": 0.5 / MILLION, "output_token": 1.5 / MILLION},
    "gpt-4-turbo": {"input_token": 10 / MILLION, "output_token": 30 / MILLION},
    "claude-haiku": {"input_token": .25 / MILLION, "output_token": 1.25 / MILLION},
    "claude-sonnet": {"input_token": 3 / MILLION, "output_token": 15 / MILLION},
    "claude-opus": {"input_token": 15 / MILLION, "output_token": 75 / MILLION},
    "gemini-nano": {"input_token": None, "output_token": None},  
    "gemini-pro": {"input_token": 4 * 0.000125 / THOUSAND, "output_token": 4 * 0.000375 / THOUSAND},  # Need to convert pricing from characters to tokens: 1 tok ~ 4 chars
    "gemini-ultra": {"input_token": None, "output_token": None},
    "gemini-1.5-pro": {"input_token": 4 * 0.00125 / THOUSAND, "output_token": 4 * 0.00375 / THOUSAND},
    "gemini-1.5-flash": {"input_token": 4 * 0.000125 / THOUSAND, "output_token": 4 * 0.000375 / THOUSAND},
    "palm-2": {"input_token": 4 * 0.00020 / THOUSAND, "output_token": 4 * 0.0004 / THOUSAND},
    "codey": {"input_token": 4 * 0.00025 / THOUSAND, "output_token": 4 * 0.0005 / THOUSAND},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+",
                        choices=["gpt-3.5-turbo", "gpt-4-turbo",
                                 "claude-haiku", "claude-sonnet", "claude-opus",
                                 "gemini-nano", "gemini-pro", "gemini-ultra", "gemini-1.5",
                                 "palm-2", "codey"])
    parser.add_argument("--input_len", type=int)
    parser.add_argument("--output_len", type=int)
    parser.add_argument("--num_queries", type=int, default=1)
    args = parser.parse_args()
    
    models = args.model
    
    for model in models:
        model_token = model_tokens[model]
        input_cost = args.input_len * model_token["input_token"]
        output_cost = args.output_len * model_token["output_token"]
        query_cost = input_cost + output_cost
        experiment_cost = args.num_queries * query_cost
        
        print("Model: ", model)
        print(f"Experiment cost: ${experiment_cost:.3f}")
        print(f"Query cost: ${query_cost:.3f}")
        print(f"Input cost: ${input_cost:.3f}")
        print(f"Output cost: ${output_cost:.3f}")
        print(25*"#")