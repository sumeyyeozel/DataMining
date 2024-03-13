from transformers import AutoTokenizer, AutoModelForCausalLM
from evalplus.data import get_human_eval_plus, write_jsonl
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B")
model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")

def GEN_SOLUTION(problem: dict) -> dict:
    task_id = problem['task_id']
    prompt = problem['prompt']
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    result = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_beams=4, num_return_sequences=10)
    solutions = [tokenizer.decode(res, skip_special_tokens=True) for res in result]
    samples = [{"task_id": task_id, "solution": sol} for sol in solutions]
    return samples


samples = [
        {"task_id": task_id, "solution": GEN_SOLUTION(problem["prompt"])}
        for task_id, problem in get_human_eval_plus().items()
    ]

write_jsonl("samples.jsonl", samples)