prompts = {
    "solve_gsm8k": "Solve the following question thinking step by step. Do NOT use code.\n\n{question}",
    "eval_gsm8k_single_step": "Is the following final answer to the question correct? Do not use code. Just give a one word yes/no answer.\n\n\
{prompt}{model_answer}",
    "eval_gsm8k_step_by_step": "Is the following final answer to the question correct? Do not use code. Check each step one at a time, step by step. \
Then use your reasoning at each step to decide on correctness of the final answer. Then give your final verdict at the END of your response.\n\n\
{prompt}{model_answer}",
    "eval_gsm8k_default": "Is the following final answer to the question correct? Do not use code. Do not give your final verdict until after any justification.\n\n\
{prompt}{model_answer}",
    "eval_gsm8k_deliberated": "Is the following final answer to the question correct? Do not use code. \
Do not give your final verdict until AFTER any justification. Give arguments for both why the solution may be correct and why it may be incorrect. \
Then give your final verdict.\n\n\
{prompt}{model_answer}",
    "eval_gsm8k_ref_sol": "Is the following final answer to the question correct? \
You will be given an accompanying reference solution to use. Do not use code. \
Do not give your final verdict until after any justification.\
For your final verdict write 'Final Verdict: '.\n\n\
{prompt}\n\n\
Ref: {model_answer_1}\n\n\
A: {model_answer_2}",
    "refine_gsm8k": "Consider the following math problems.\n\n\
{prompt}\n\n\
Additionally consider the initial solution draft below.\n\n\
{model_answer}\n\n\
Now considering both the question and solution draft, generate a refinement fixing any errors you see \
or correct solutions you can think of. Do NOT use code. Make sure to write your final answer at the end of your solution \
using the identifier 'Final Answer : '.",
    "eval_gsm8k_draft_refine": "You will be shown multiple candidate solutions to a math question and \
must decide which are Correct or Incorrect. You are encouraged to compare and contrast both to improve your decision. Do NOT use code. \
Do not give your final verdict until after any justification. \
For your final verdict for A1 write 'Final Verdict A1: ' and similarly for A2.\n\n\
{prompt}\n\n\
A1: {model_answer}\n\n\
A2: {refinement}",
    "eval_gsm8k_two_sol": "You will be shown multiple candidate solutions to a math question and \
must decide which are Correct or Incorrect. You are encouraged to compare and contrast both to improve your decision. Do NOT use code. \
Do not give your final verdict until after any justification. \
For your final verdict for A1 write 'Final Verdict A1: ' and similarly for A2.\n\n\
{prompt}\n\n\
A1: {model_answer_1}\n\n\
A2: {model_answer_2}",
    "eval_gsm8k_three_sol": "You will be shown multiple candidate solutions to a math question and \
must decide which are Correct or Incorrect. You are encouraged to compare and contrast both to improve your decision. Do NOT use code. \
Do not give your final verdict until after any justification. \
For your final verdict for A1 write 'Final Verdict A1: ' and similarly for A2, A3.\n\n\
{prompt}\n\n\
A1: {model_answer_1}\n\n\
A2: {model_answer_2}\n\n\
A3: {model_answer_3}",
    "eval_rerank_five_sol": "You will be shown multiple candidate solutions to a math question and \
must decide which is most likely to be correct. You are encouraged to compare and contrast both to improve your decision. Do NOT use code. \
Do not give your final verdict until after any justification. \
For your final verdict write 'Final Verdict:  A1' if you think A1 is most likely to be correct or similarly for A2, A3.\n\n\
{prompt}\n\n\
A1: {model_answer_1}\n\n\
A2: {model_answer_2}\n\n\
A3: {model_answer_3}\n\n\
A4: {model_answer_4}\n\n\
A5: {model_answer_5}",
    "prm_verifier_default": "Is the following final answer to the question correct? Do not use code. \
For each step write an intermediate verdict as IV: positive if the step is correct, 'negative; if incorrect, and 'neutral' if unsure. \
Then give your final verdict at the END of your response as 'FV: positive' if the solution is correct and 'FV: negative' otherwise.\n\n\
Your full analysis should look like \
STEP_0, Discussion, IV: [verdict]\n\
...\n\
STEP_L, Discussion, IV: [verdict]\n\
Discussion, FV: [verdict]\n\
{prompt}{model_answer}",
    "eval_dense_two_sol": "You will be shown multiple candidate solutions to a math problem and must assess each step \
to determine correctness of the final answer. You are encouraged to compare and contrast both to improve your decision. Do NOT use code. \
Do not give your final verdict until after any justification. \
For each step write an intermediate verdict as IV: positive if the step is correct, 'negative; if incorrect, and 'neutral' if unsure. \
Then give your final verdict at the END of your response as 'FV: positive' if the solution is correct and 'FV: negative' otherwise.\n\n\
Your full analysis should look like \
A1: STEP_0, Discussion, IV1: [verdict]\n\
...\n\
STEP_L, Discussion, IV1: [verdict]\n\
A2: STEP_0,...\
Discussion, FV1: [verdict], FV2: [verdict]\n\
{prompt}\n\n\
A1: {model_answer_1}\n\n\
A2: {model_answer_2}",
}