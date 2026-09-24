import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name: str):
    print(f"Attempting to load {model_name} with 8-bit (bitsandbytes) if available...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Try 8-bit load (requires bitsandbytes + compatible GPUs)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True,
        )
        print("Loaded model in 8-bit mode.")
        return tokenizer, model
    except Exception as e:
        print("8-bit load failed or unsupported:", e)

    # Try fp16 on GPU
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print("Loaded model in fp16 mode.")
        return tokenizer, model
    except Exception as e:
        print("fp16 load failed:", e)

    # Fallback to CPU (fp32)
    print("Falling back to CPU (this will be slow and may OOM for large models).")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "codellama/CodeLlama-7b-hf"))
    parser.add_argument("--prompt", type=str, help="One-shot prompt to generate a response for")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive prompt loop")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)

    prompt = (
        "# Task\nWrite a concise function in Python that returns the nth Fibonacci number efficiently.\n# Solution\n"
    )

    def generate_text(text):
        inputs = tokenizer(text, return_tensors="pt")
        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except StopIteration:
            pass

        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        return tokenizer.decode(gen[0], skip_special_tokens=True)

    if args.interactive:
        print("Interactive mode — type a prompt and press Enter. Type 'exit' or 'quit' to stop.")
        try:
            while True:
                user_in = input("You: ").strip()
                if not user_in:
                    continue
                if user_in.lower() in ("exit", "quit"):
                    print("Exiting interactive mode.")
                    break
                resp = generate_text(user_in)
                print("Model:", resp)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
        return

    if args.prompt:
        out_text = generate_text(args.prompt)
        print('\n' + out_text)
        return

    # default behavior: run the built-in demo prompt
    out = generate_text(prompt)
    print('\n' + out)


if __name__ == '__main__':
    main()
