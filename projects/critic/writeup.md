# Evaluating the zero-shot verification abilities of GPT-4 on math problems

Recent work [[1]](#references) studies the ability of SOTA large language models to reliably refine incorrect answers to math problems without any external feedback. Correctly refining a solution draft $D$ of a question $Q$ is a difficult task, requiring the model to first decide *when* to refine and then *where* and *how* to refine. SOTA models struggle to reliably decide when to refine incorrect drafts, immediately limiting refinement performance. Working mathematicans are adept at identifying reasoning errors and gauging the expected outcome of a proof strategies. Thus improving the abilitiy of an LLM to act as a zero-shot verifier of reasoning traces is crucial for advanced understanding of mathematics. 

In this work we examine the zero-shot abilities of LLMs to generate and *verify* correctness of candidate solutions in GSM8K [[2]](#references) and MATH [[3]](#references). We primarily consider verifying final answer correctness, but additionally evaluate the ability of of LLMs to identify step level errors on solutions to problems in the MATH dataset using labels released in [[4]](#references). We benchmark several prompting strategies including single-token verification, Chain of Thought (**CoT**), and debate. We then propose a new strategy we call **in-context multi-sample classification**. This new method encourages the verifier to compare and contrast multiple diverse candidate solutions to the same question in-context, leading to improved performance in some cases.

## Setup


## Results


## References

[1]: Cite refining failure paper

[2]: Cite GSM8K

[3]: Cite math

[4]: Cite PRM paper