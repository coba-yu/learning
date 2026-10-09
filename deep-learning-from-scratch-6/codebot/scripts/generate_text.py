from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device


if __name__ == "__main__":
    tokenizer_path = "data/tiny_codes.pkl"
    model_path = "models/model_pretrain.pt"

    prompt = "def"
    max_new_tokens = 200
    temperature = 1.0

    device = get_device()

    tokenizer = BPETokenizer.load_from(tokenizer_path)
    model = GPT.load_from(model_path, device=device)

    for i in range(5):
        print(f"--- Sample {i+1} ---")
        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print(generated_text)
        print()
