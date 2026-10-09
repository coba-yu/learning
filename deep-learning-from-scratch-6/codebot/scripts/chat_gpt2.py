from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device


def format_prompt(user_message: str) -> str:
    return f"### Instruction:\n{user_message}\n\n### Response:\n"


if __name__ == "__main__":
    model_path = "models/model_sft.pt"
    tokenizer_path = "data/tiny_codes.pkl"
    max_new_tokens = 200
    temperature = 1.0

    device = get_device()

    tokenizer = BPETokenizer.load_from(tokenizer_path)
    model = GPT.load_from(model_path, device=device)

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        prompt = format_prompt(user_input)
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        if "\n" in response:
            print(f"Bot:\n{response}")
        else:
            print(f"Bot: {response}")
