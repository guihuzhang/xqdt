#Imports
import argparse
import json
import multiprocessing as mp
import os
import random
import re
from typing import List

import numpy as np
from tqdm import tqdm


def configure_multiprocessing():
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def get_output_filename(args, suffix):
    if args.output_filename:
        filename = f"{args.output_filename}_{args.fewshot_style}_{args.generation_technique}_{args.descriptions}_{args.labels}_{args.target_lang}.{suffix}"
    else:
        input_stem = os.path.splitext(os.path.basename(args.input_file))[0]
        filename = f"{input_stem}_{args.fewshot_style}_{args.generation_technique}_{args.descriptions}_{args.labels}_{args.target_lang}.{suffix}"
    return os.path.join(args.output_dir, filename)


def get_entry_key(entry, idx):
    if entry.get('xml_id'):
        return entry['xml_id']
    if entry.get('id_short'):
        return entry['id_short']
    return f'idx_{idx}'


def load_completed_predictions(jsonl_path):
    completed = {}
    if not os.path.exists(jsonl_path):
        return completed

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed JSONL line {line_num} in {jsonl_path}")
                continue
            if not isinstance(row, dict):
                print(
                    f"Warning: skipping non-dict JSONL line {line_num} in {jsonl_path} "
                    f"(type={type(row).__name__})"
                )
                continue
            key = row.get('key')
            if key is None:
                continue
            completed[key] = row
    return completed


def append_prediction(jsonl_path, row):
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def prediction_to_lexicalisation(prediction, lang_code):
    if type(prediction) == dict:
        return {lang_code: [{"comment": "", "lex": prediction['full-text']}]}
    return {lang_code: [{"comment": prediction, "lex": ""}]}


def _coerce_prediction_dict(parsed):
    if not isinstance(parsed, dict):
        return None
    for key in ("full-text", "full_text", "text"):
        value = parsed.get(key)
        if isinstance(value, str):
            return {"full-text": value.strip()}
    return None


def parse_prediction(prediction):
    text = str(prediction).strip()

    candidates = [text]
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue

        parsed_dict = _coerce_prediction_dict(parsed)
        if parsed_dict is not None:
            return parsed_dict

        if isinstance(parsed, str):
            return {"full-text": parsed.strip()}

    json_field_match = re.search(
        r'"full-text"\s*:\s*"(?P<text>(?:\\.|[^"\\])*)"',
        text,
        re.DOTALL,
    )
    if json_field_match:
        captured = json_field_match.group("text")
        try:
            captured = json.loads(f'"{captured}"')
        except Exception:
            pass
        return {"full-text": str(captured).strip()}

    plain_field_match = re.search(
        r'full-text\s*[:=]\s*(?P<text>.+?)\s*\}?$',
        text,
        re.DOTALL,
    )
    if plain_field_match:
        captured = plain_field_match.group("text").strip(' "\':\n')
        if captured:
            return {"full-text": captured}

    return {"full-text": text}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_vllm(args, model_id):
    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError("vLLM is required. Install it with `pip install vllm`.") from exc

    cache_dir = os.path.join(args.model_cache, "vllm")
    os.makedirs(cache_dir, exist_ok=True)

    kwargs = {
        "model": model_id,
        "tokenizer": model_id,
        "dtype": args.dtype,
        "download_dir": cache_dir,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs

    return LLM(**kwargs)


def build_sampling_params(args, tokenizer):
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError("vLLM is required. Install it with `pip install vllm`.") from exc

    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    extra_stop_tokens = []
    if args.model_type.startswith('llama3'):
        extra_stop_tokens.append("<|eot_id|>")
    if args.model_type in ['qwen2.5', 'qwen3']:
        extra_stop_tokens.append("<|im_end|>")

    for token in extra_stop_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in stop_token_ids:
            stop_token_ids.append(token_id)

    return SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        stop_token_ids=stop_token_ids,
    )


def render_prompt(args, tokenizer, messages):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if args.disable_thinking:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def maybe_print_prompt_tokens(args, llm, pending):
    if not args.debug_prompt_tokens or getattr(args, "_printed_prompt_tokens", False) or not pending:
        return

    tokenizer = llm.get_tokenizer()
    prompt = pending[0]["prompt"]
    ids = tokenizer(prompt, add_special_tokens=False, return_attention_mask=False).input_ids
    count = args.debug_prompt_token_count
    tokens = tokenizer.convert_ids_to_tokens(ids[:count])
    print(f"[Debug] First prompt token count: {len(ids)}")
    print(f"[Debug] First prompt token ids[:{count}]: {ids[:count]}")
    print(f"[Debug] First prompt tokens[:{count}]: {tokens}")
    print(f"[Debug] First prompt prefix: {prompt[: args.debug_prompt_char_count]!r}")
    args._printed_prompt_tokens = True


def build_messages(args, pb, entry):
    if args.generation_technique == 'dir' or len(entry['modifiedtripleset']) == 1:
        return pb.build_prompt_dir(entry)
    if args.generation_technique == 'cot':
        return pb.build_prompt_cot(entry)
    raise ValueError("vLLM simple batch generation currently supports dir and cot only.")


def generate_batch(llm, sampling_params, request_items):
    prompts = [item["prompt"] for item in request_items]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    predictions = []
    for output in outputs:
        text = output.outputs[0].text
        predictions.append(parse_prediction(text))
    return predictions


def save_data(args, data):
    output_file_path = get_output_filename(args, 'json')
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"File saved successfully to {output_file_path}.")


def main(args):
    from load_model import resolve_model_id

    configure_multiprocessing()
    set_seed(args.seed)
    model_id = resolve_model_id(args)
    llm = load_vllm(args, model_id)
    tokenizer = llm.get_tokenizer()
    sampling_params = build_sampling_params(args, tokenizer)

    from build_prompts import PromptBuilder
    pb = PromptBuilder(args)

    print('Loading Data...')
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    print('Data Loaded...')

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = get_output_filename(args, 'jsonl')
    completed_predictions = load_completed_predictions(jsonl_path)
    print(f'Loaded {len(completed_predictions)} completed predictions from {jsonl_path}')

    print('Beginning Generations...')
    pending: List[dict] = []
    loop = tqdm(enumerate(data['entries']), total=len(data['entries']))
    for idx, entry in loop:
        entry_key = get_entry_key(entry, idx)

        if entry_key in completed_predictions:
            restored_prediction = completed_predictions[entry_key]['prediction']
            entry['lexicalisations'] = prediction_to_lexicalisation(restored_prediction, args.lang_code)
            loop.set_postfix_str(f"resume {entry_key}")
            continue

        messages = build_messages(args, pb, entry)
        prompt = render_prompt(args, tokenizer, messages)
        pending.append(
            {
                "idx": idx,
                "entry": entry,
                "key": entry_key,
                "messages": messages,
                "prompt": prompt,
            }
        )

        if len(pending) >= args.batch_size:
            flush_pending(args, llm, sampling_params, pending, completed_predictions, jsonl_path, loop)
            pending = []

    if pending:
        flush_pending(args, llm, sampling_params, pending, completed_predictions, jsonl_path, loop)

    return data


def flush_pending(args, llm, sampling_params, pending, completed_predictions, jsonl_path, loop):
    maybe_print_prompt_tokens(args, llm, pending)
    predictions = generate_batch(llm, sampling_params, pending)
    for item, prediction in zip(pending, predictions):
        entry = item["entry"]
        entry_key = item["key"]

        if args.do_print_check:
            for message in item["messages"]:
                print(message["role"])
                print(message['content'])
            print("Prediction:\n", prediction)

        entry['lexicalisations'] = prediction_to_lexicalisation(prediction, args.lang_code)
        row = {
            'key': entry_key,
            'index': item["idx"],
            'lang': args.lang_code,
            'backend': 'vllm',
            'prompt_mode': 'simple_text',
            'model': args.model_id_or_path or args.model_type + "-" + args.model_size,
            'disable_thinking': args.disable_thinking,
            'prediction': prediction,
        }
        append_prediction(jsonl_path, row)
        completed_predictions[entry_key] = row
        loop.set_postfix_str(f"saved {entry_key}")


if __name__ == '__main__':
    #Main Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='path/to/file')
    parser.add_argument('--fewshot_source', type=str, default='path/to/file')
    parser.add_argument('--output_dir', type=str, default='path/to/output_dir')
    parser.add_argument('--model_cache', type=str, default='path/to/dir')
    parser.add_argument('--hf_token', type=str, default='')
    parser.add_argument('--model_source', type=str, default='hf', help='modelscope or hf')
    parser.add_argument('--model_id_or_path', type=str, default='', help='Optional model id or local path. Overrides model_source/model_type/model_size mapping.')
    parser.add_argument('--model_size', type=str, default='8B', help='0.6B, 1.7B, 4B, 7B, 8B, 14B, 32B, 70B, 72B, or 405B')
    parser.add_argument('--model_type', type=str, default='llama3', help='llama3, qwen2.5, or qwen3')
    parser.add_argument('--target_lang', type=str, default='en', help='en, ru, es, zh, ar, or fr')

    #Optional Arguments
    parser.add_argument('--output_filename', type=str, default='')
    parser.add_argument('--lang_code', type=str, default='')
    parser.add_argument('--fewshot_lang', type=str, default='en')
    parser.add_argument('--fewshot_style', type=str, default='overlap', help='overlap, size, or random')
    parser.add_argument('--prompt_template', type=str, default='current', help='current or mucal')
    parser.add_argument('--generation_technique', type=str, default='dir', help='dir, cml, or cot')
    parser.add_argument('--descriptions', type=str, default='all', help='all, properties, or none')
    parser.add_argument('--labels', type=bool, default=False)
    parser.add_argument('--num_fewshot', type=int, default=4)
    parser.add_argument('--do_print_check', type=bool, default=False)
    parser.add_argument('--disable_thinking', action='store_true', help='Pass enable_thinking=False to chat templates that support it, e.g. Qwen3.')
    parser.add_argument('--debug_prompt_tokens', action='store_true', help='Print token ids/tokens for the first rendered prompt.')
    parser.add_argument('--debug_prompt_token_count', type=int, default=16)
    parser.add_argument('--debug_prompt_char_count', type=int, default=240)

    # vLLM Arguments
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_model_len', type=int, default=None)
    parser.add_argument('--quantization', type=str, default='')
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--enforce_eager', action='store_true')
    parser.add_argument('--max_num_seqs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)

    #Generation Arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_new_tokens', type=int, default=1280)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    args = parser.parse_args()

    language_dict = {'br':'Breton', 'cy':'Welsh', 'en': 'English', 'fr':'French', 'ga':'Irish', 'it':'Italian', 'mt':'Maltese', 'pt':'Portuguese', 'ru': 'Russian', 'es': 'Spanish', 'zh': 'Simplified Chinese',}

    args.target_language = language_dict[args.target_lang]
    args.fewshot_language = language_dict[args.fewshot_lang]

    if args.lang_code == '':
        args.lang_code = args.target_lang

    #Generate Lexicalisations
    data = main(args)

    #Save the Output
    save_data(args, data)
