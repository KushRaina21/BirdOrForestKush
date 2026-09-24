#!/usr/bin/env python3
import argparse
import os
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description='Download a file (e.g. GGUF) from a Hugging Face repo')
parser.add_argument('--repo_id', required=True, help='Hugging Face repo id (owner/repo)')
parser.add_argument('--filename', required=True, help='Filename in the repo to download (e.g. model.gguf)')
parser.add_argument('--dest', default=os.path.expanduser('~/models'), help='Destination directory')
args = parser.parse_args()

os.makedirs(args.dest, exist_ok=True)
print(f"Downloading {args.filename} from {args.repo_id} to {args.dest} (may require HF login/acceptance)...")
path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, cache_dir=args.dest)
print('Downloaded to:', path)
print('You can then run:')
print(f"python3 run_llama_cpp.py --model_path {path} --prompt \"Explain recursion in two lines\"")
