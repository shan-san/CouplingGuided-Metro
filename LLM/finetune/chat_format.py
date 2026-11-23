"""One chat-template mode for JSON-guidance training and generation."""


def render_chat(tokenizer, messages, *, generation=False):
    return tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=generation, enable_thinking=False)
