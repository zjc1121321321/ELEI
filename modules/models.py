"""LLM backends. Credentials are read only when the selected backend is called."""

import os

from config import LLM_CONFIG


def get_backend_config(provider=None):
    provider = provider or LLM_CONFIG["provider"]
    if provider not in LLM_CONFIG["backends"]:
        choices = ", ".join(LLM_CONFIG["backends"])
        raise ValueError(f"Unknown LLM provider {provider!r}; choose from: {choices}")
    return provider, LLM_CONFIG["backends"][provider]


def query_model(prompt, debug=True, provider=None):
    provider, settings = get_backend_config(provider)
    if settings["language"] == "zh":
        system = "你是一个符号回归专家，擅长根据数据信息和样本点生成数学表达式。具体要求参考prompt内容。"
    elif provider == "openai":
        system = "You are a helpful assistant,Please refer to the prompt for specific requirements."
    else:
        system = (
            "You are a symbolic regression expert, skilled at generating mathematical "
            "expressions based on data information and sample points. "
            "Please refer to the prompt for specific requirements."
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    if debug:
        print("========== Prompt ==========\n" + prompt + "\n============================")

    if provider == "llama3":
        import requests

        response = requests.post(
            settings["base_url"].rstrip("/") + "/api/chat",
            json={"model": settings["model"], "stream": False, "messages": messages},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    key_name = settings["api_key_env"]
    api_key = os.getenv(key_name)
    if not api_key:
        raise RuntimeError(f"Set {key_name} before calling the {provider} backend.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=settings["base_url"])
    response = client.chat.completions.create(
        model=settings["model"], messages=messages, stream=False,
    )
    output = response.choices[0].message.content
    if not output or not output.strip():
        raise RuntimeError(f"The {provider} backend returned an empty response.")
    return output


def query_deepseek(prompt, debug=True):
    return query_model(prompt, debug=debug, provider="deepseek")


def query_qwen(prompt, debug=True):
    return query_model(prompt, debug=debug, provider="qwen")


def query_gpt(prompt, debug=True):
    return query_model(prompt, debug=debug, provider="openai")


def query_llama3(prompt, debug=True):
    return query_model(prompt, debug=debug, provider="llama3")
