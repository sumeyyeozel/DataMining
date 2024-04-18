from transformers import AutoTokenizer, AutoModelForCausalLM
from dm.evalplus import get_human_eval_plus, write_jsonl
import torch
import time
import matplotlib.pyplot as plt

start = time.time()
start_cpu = time.process_time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Computing device: {device}")

tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B")
model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")

def GEN_SOLUTION(prompt: str) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    max_length = 50  # maximum length for the generated sequences
    num_beams = 1  #  number of beams to 1 for only one solution
    no_repeat_ngram_size = 2  # size of n-grams to avoid repeating
    result = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length + input_ids.size(1),
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_return_sequences=1  # Only one solution per prompt
    )
    solution = tokenizer.decode(result[0], skip_special_tokens=True)
    return solution

samples = [
    {"task_id": task_id, "solution": GEN_SOLUTION(problem["prompt"])}
    for task_id, problem in get_human_eval_plus().items()
]


# runtime measurements

if torch.cuda.is_available(): torch.cuda.synchronize()
end = time.time()
end_cpu = time.process_time()

clock_t = end-start
cpu_t = end_cpu-start_cpu
print("Wall-clock time [s]: ",clock_t)
print("CPU time [s]: ",cpu_t)


# Ensure each solution is a string, not a list
for sample in samples:
    sample["solution"] = str(sample["solution"])

write_jsonl("samples.jsonl", samples)


fig, ax = plt.subplots()
plt.bar(["CPU time", "Wall-clock time"], [cpu_t, clock_t])
ax.set_ylabel('Time (s)')
ax.set_title('Sample generation')
plt.savefig('runtime.png')


