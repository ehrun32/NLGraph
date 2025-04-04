# check_output.py
import numpy as np

path = "log/cycle/gpt-3.5-turbo-medium-20250404---17-29-CoT+SC/answer.npy"
answer = np.load(path, allow_pickle=True)
print("Answer:", answer)

