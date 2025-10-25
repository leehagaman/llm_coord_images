import numpy as np
from vllm import SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

def get_prob(llm, tokenizer, system_prompt_string, user_prompt_strings, answer_start_string, answer_end_strings, print_all_probs=False, print_all_answer_probs=False):

    all_softmax_probs = []

    for user_prompt_string in tqdm(user_prompt_strings):

        message_no_answer = [
            {"role": "system", "content": system_prompt_string},
            {"role": "user", "content": user_prompt_string},
            {"role": "assistant", "content": answer_start_string},
        ]
        no_answer_prompt_string_with_end_str = tokenizer.apply_chat_template(
            message_no_answer,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        no_answer_prompt_string_split_by_end = no_answer_prompt_string_with_end_str.split("<|im_end|>")
        no_answer_prompt_string = ""
        for part in no_answer_prompt_string_split_by_end[:-1]:
            no_answer_prompt_string += part + "<|im_end|>"
        num_no_answer_tokens = len(tokenizer.encode(no_answer_prompt_string)) - 1

        curr_prompt_logprobs = []

        for answer_end_string in answer_end_strings:

            messages = [
                {"role": "system", "content": system_prompt_string},
                {"role": "user", "content": user_prompt_string},
                {"role": "assistant", "content": answer_start_string + answer_end_string},
            ]

            prompt_string = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )

            sampling_params = SamplingParams(
                max_tokens=1,
                logprobs=1,       
                temperature=0.0,  # Use 0.0 for deterministic, high-probability output
                prompt_logprobs=1,
            )

            outputs = llm.generate(prompt_string, sampling_params, use_tqdm=False)

            output = outputs[0]

            prompt_token_ids = output.prompt_token_ids
            prompt_logprobs = output.prompt_logprobs

            total_answer_logprob = 0.

            for token_num in range(num_no_answer_tokens, len(prompt_token_ids) - 1):
                token_id = prompt_token_ids[token_num]
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                logprob_dict = prompt_logprobs[token_num]

                logp = next(iter(logprob_dict.values())).logprob
                total_answer_logprob += logp
                prob = np.exp(logp)
                if print_all_answer_probs:
                    print(f"token_num: {token_num}, {token_str}, {logp}, {prob}")

            #print(f"Probability for answer '{answer_end_string}': {np.exp(total_answer_logprob)}")

            curr_prompt_logprobs.append(total_answer_logprob)

            if print_all_probs:

                print("Unified Token Analysis (Prompt + Generated):")
                print("-" * 100)
                print("{:<10} {:<8} {:<8} {:<20} {:<12} {:<10}".format("Source", "Pos", "ID", "Token", "LogP", "Prob"))
                print("-" * 100)

                pos = 1
                # Stream prompt tokens first
                for i, (token_id, logprob_dict) in enumerate(zip(prompt_token_ids, prompt_logprobs)):
                    logp = None
                    prob = None
                    if logprob_dict is not None and len(logprob_dict) > 0:
                        logp = next(iter(logprob_dict.values())).logprob
                        prob = float(np.exp(logp))

                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)

                    print("{:<10} {:<8} {:<8} {:<20} {:<12} {:<10}".format(
                        "PROMPT",
                        pos,
                        token_id,
                        repr(token_str),
                        f"{logp:.4f}" if logp is not None else "N/A",
                        f"{prob:.4f}" if prob is not None else "N/A",
                    ))
                    pos += 1

                # Then stream generated tokens
                generated_tokens = output.outputs[0].token_ids
                generated_token_logprobs = output.outputs[0].logprobs
                for i, (token_id, logprob_dict) in enumerate(zip(generated_tokens, generated_token_logprobs)):
                    logp = next(iter(logprob_dict.values())).logprob
                    prob = float(np.exp(logp))
                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)

                    print("{:<10} {:<8} {:<8} {:<20} {:<12.4f} {:<10.4f}".format(
                        "GENERATED",
                        pos,
                        token_id,
                        repr(token_str),
                        logp,
                        prob,
                    ))
                    pos += 1

                print("-" * 100)

        # softmax the curr_prompt_logprobs
        curr_prompt_probs = np.exp(curr_prompt_logprobs)
        curr_prompt_probs = curr_prompt_probs / np.sum(curr_prompt_probs)
        all_softmax_probs.append(curr_prompt_probs)

    return all_softmax_probs

