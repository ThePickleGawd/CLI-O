import re, json, pipes

def try_parse_tool_calls(content: str):
    """Try parse the tool calls."""
    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            tool_calls.append({"type": "function", "function": func})
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as e:
            print(f"\n❌ Failed to parse tool call: {e}\nContent: {m.group(1)}")
    if tool_calls:
        prefix = content[:offset].strip()
        return {
            "role": "assistant",
            "content": prefix if prefix else "",
            "tool_calls": tool_calls
        }
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}


def convert_to_single_line(code_snippet: str) -> str:
    # from https://stackoverflow.com/a/68203945

    regex = r"^(.+?)#\sBEGIN\s#.+?#\sEND\s#(.+)$"
    core = re.sub(regex, "\\1\\2", code_snippet, 0, re.MULTILINE | re.DOTALL)
    escaped = f"{core!r}"
    outer = pipes.quote(f"exec({escaped})")
    oneline = f"python -c {outer}"
    return oneline