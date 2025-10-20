import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print_token_debug = False

model_name = "Qwen/Qwen3-8B"

if not torch.cuda.is_available():
    raise ValueError("CUDA is not available")

device = "cuda"
dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=dtype,
    device_map="auto",
)

pre_prompt = "What is"
variations = [
    " 1+1?",
    " 2+1?",
    " 3+1?",
    " 4+1?",
]

messages_sys = [{"role": "system", "content": "You are a helpful assistant, who answers immediately without any thinking out loud or explanation."}]
messages_user_pre = [{"role": "user", "content": pre_prompt}]
messages_pre = messages_sys + messages_user_pre

pre_ids = tokenizer.apply_chat_template(
    messages_pre,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
    enable_thinking=False,
)
pre_ids = pre_ids[:, :-2] # removing the <|im_end|> token, since the user input isn't finished yet
pre_ids = pre_ids.to(device)

if print_token_debug:
    print("pre_ids:")
    for pre_id in pre_ids[0]:
        print(tokenizer.decode([pre_id], skip_special_tokens=False), end="")
    print("\ndone with pre_ids")

with torch.no_grad():
    prefix_out = model(input_ids=pre_ids, use_cache=True, output_hidden_states=True)
    prefix_cache = prefix_out.past_key_values  # KV cache for system prefix
    prefix_hidden = prefix_out.hidden_states[-1]  # optional, if you want prefix embedding

for var in variations:
    print("--------------------------------")
    messages_user_var = [{"role": "user", "content": pre_prompt + var}]
    messages_assistant = [{"role": "assistant", "content": "The answer is"}]
    messages_var = messages_sys + messages_user_var + messages_assistant

    if print_token_debug:
        print("current messages_var:")
        print("   ", messages_var)
    full_ids = tokenizer.apply_chat_template(
        messages_var,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )

    # removing the <|im_end|> token, since the assistant answer isn't finished yet
    full_ids = full_ids[:, :-2] 
    # adding a trailing space, which should be present before a numerical answer
    full_ids = torch.cat([full_ids, torch.tensor([[220]])], dim=1)
    full_ids = full_ids.to(device)

    if print_token_debug:
        print("full_ids:")
        for full_id in full_ids[0]:
            print(tokenizer.decode([full_id], skip_special_tokens=False), end="")
        print("\ndone with full_ids")
    suffix_ids = full_ids[:, pre_ids.shape[-1]:]

    with torch.no_grad():
        # Reuse cached system prefix via KV cache
        out = model(input_ids=suffix_ids, past_key_values=prefix_cache, output_hidden_states=True)

    # hidden state before logits
    hidden = out.hidden_states[-1]          # [1, seq_len, hidden_dim]
    logits = model.lm_head(hidden)          # [1, seq_len, vocab_size]
    # Use float32 for softmax to avoid underflow
    next_logits = logits[:, -1, :].to(torch.float32)
    probs = torch.softmax(next_logits, dim=-1)  # probs for last token


    top_5_token_ids = []
    top_prob, top_id = torch.topk(probs[0], k=5)
    for i in range(5):
        top_5_token_ids.append(top_id[i].item())

    print("top_5_token_texts:")
    for token_id in top_5_token_ids:
        print(f"'{tokenizer.decode([token_id], skip_special_tokens=False)}'", end="")
    print()

    integer_token_ids = []
    for token_id in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        integer_token_ids.append(tokenizer.encode(token_id, add_special_tokens=False)[0])

    end_token_ids = [13, 151645]

    print("integer token ids:")
    for token_id in integer_token_ids:
        print(f"'{tokenizer.decode([token_id], skip_special_tokens=False)}'", end="")
    print()

    candidate_token_ids = list(set(top_5_token_ids + integer_token_ids))

    candidate_token_sequences = [[candidate_token_id] + [13, 151645] for candidate_token_id in candidate_token_ids]

    print("candidate_token_sequences:")
    for candidate_token_sequence in candidate_token_sequences:
        print(f"'{tokenizer.decode(candidate_token_sequence, skip_special_tokens=False)}'", end="")
    print()


    for candidate_token_id in candidate_token_ids:
        pass

        # Now, calculate the probability of a period immediately after this answer, running the model again
        

    best_text, best_id = max(candidate_ids, key=lambda ti: probs[0, ti[1]].item())
    print(f"P('{best_text}' | prefix + '{var[:20]}...') = {probs[0, best_id].item():.3e}")

