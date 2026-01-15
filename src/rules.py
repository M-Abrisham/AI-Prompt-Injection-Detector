import re


RULES = [
    ("override", r"ignore (all|any|previous) (instructions|rules)"),
    ("system_prompt", r"(reveal|show|print).{0,40}(system prompt|developer message)"),
    ("roleplay", r"\byou are now\b|\bpretend to be\b|\bact as\b"),
    ("secrecy", r"don't tell (the user|anyone)|keep this secret"),
    ("exfiltration", r"\b(upload|send|exfiltrate|forward)\b.{0,60}\b(to|http|https|email)\b"),
]

def rule_score(text: str):
    t = text.lower()
    hits = []
    for name, pattern in RULES:
        if re.search(pattern, t):
            hits.append(name)
    return hits
