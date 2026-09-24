from llama_cpp import Llama
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, help='Path to GGUF model file (e.g. /path/to/qwen-2.5-coder.gguf)')
parser.add_argument('--prompt', default='Write a short summary of recursion in Python', help='Prompt to send to model')
parser.add_argument('--max_tokens', type=int, default=256)
args = parser.parse_args()

llm = Llama(model_path=args.model_path)
resp = llm.create(prompt=args.prompt, max_tokens=args.max_tokens)
print(resp['choices'][0]['text'])
