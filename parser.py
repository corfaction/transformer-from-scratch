import json
import sys

def parse_telegram_chat(chat_path):
    with open(chat_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    odd = True
    output_lines = []
    
    def is_valid_text(text):
        return text and isinstance(text, str) and text.strip()
    
    def add_text(text):
        nonlocal odd
        if is_valid_text(text):
            if odd:
                output_lines.append(text.strip())
            odd = not odd
    
    def extract(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'text':
                    if isinstance(v, str):
                        add_text(v)
                    elif isinstance(v, list):
                        for part in v:
                            if isinstance(part, str):
                                add_text(part)
                            elif isinstance(part, dict) and 'text' in part:
                                add_text(part['text'])
                else:
                    extract(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item)
    
    extract(data)
    return '\n'.join(output_lines)

messages = "" 

for file in range(1, len(sys.argv) - 1):
	messages += str(parse_telegram_chat(sys.argv[file]))

with open(sys.argv[-1] + "/dataset.txt", 'w', encoding='utf-8') as f:
	f.write(messages)


