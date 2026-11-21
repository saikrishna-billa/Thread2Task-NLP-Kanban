import re
import logging
import spacy
from typing import List, Tuple

for logger_name in ["transformers", "fastcoref", "urllib3", "huggingface_hub"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

nlp = spacy.load("en_core_web_sm")

GENERIC_NOUNS = {
    "issue", "issues", "problem", "problems", "thing", "things", 
    "bug", "bugs", "error", "errors", "detail", "details", "something", "anything"
}

ENVIRONMENT_WORDS = {
    "production", "prod", "staging", "dev", "development", 
    "server", "servers", "system", "systems", "environment", "cluster", "metrics"
}

TECHNICAL_ASSET_KEYWORDS = {
    "key", "keys", "api", "gateway", "database", "db", "token", "tokens",
    "service", "code", "file", "script", "asset", "assets", "log", "logs",
    "budget", "repo", "repository", "auth", "login", "app", "application",
    "credential", "credentials", "hotfix", "patch", "deck", "slides", "banner",
    "invoice", "contract", "refund", "tracker", "listener", "cluster", "nodes"
}

class GlobalAssetStore:
    """
    Persistent state buffer that tracks concrete technical nouns across 
    chunk boundaries in long (100–1000 line) threads.
    """
    def __init__(self):
        self.assets: List[str] = []
        self.seen_set = set()

    def add_asset(self, entity_str: str):
        norm = normalize_entity_casing(entity_str)
        if norm.lower() not in self.seen_set and len(norm) > 2:
            self.seen_set.add(norm.lower())
            self.assets.append(norm)

    def get_recent_assets(self, max_count: int = 5) -> List[str]:
        return self.assets[-max_count:] if self.assets else []


def is_non_target_entity(entity_str: str) -> bool:
    words = [w.lower() for w in re.findall(r'\b\w+\b', entity_str)]
    if not words:
        return True
    stop_words = GENERIC_NOUNS | ENVIRONMENT_WORDS | {"it", "this", "that", "them", "these", "those", "guys", "we", "i", "you", "a", "an", "the", "datadog"}
    return all(w in stop_words for w in words)


def clean_grammar_and_articles(text: str) -> str:
    text = re.sub(r'\b(the|a|an)\s+(the|a|an)\b', r'\2', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthe\s+([Aa])\b', r'\1', text)
    return text


def normalize_entity_casing(entity_str: str) -> str:
    cleaned = entity_str.strip()
    if cleaned.startswith("The "):
        cleaned = "the " + cleaned[4:]
    elif cleaned.startswith("A "):
        cleaned = "a " + cleaned[2:]
    return cleaned


def resolve_coreferences_chunk(
    lines: List[str], 
    asset_store: GlobalAssetStore
) -> List[str]:
    """
    Resolves pronouns within a message chunk, utilizing both intra-chunk
    and rolling cross-chunk antecedents from GlobalAssetStore.
    """
    if not lines:
        return lines

    line_entities = []

    for line in lines:
        doc = nlp(line)
        current_line_assets = []
        for chunk in doc.noun_chunks:
            cleaned = re.sub(r'^[A-Za-z0-9_]+:\s*', '', chunk.text.strip()).strip()
            if cleaned and not is_non_target_entity(cleaned) and len(cleaned) > 2:
                norm = normalize_entity_casing(cleaned)
                current_line_assets.append(norm)
                asset_store.add_asset(norm)
        line_entities.append(current_line_assets)

    resolved_lines = []
    for line_idx, line in enumerate(lines):
        resolved_line = line
        
        # Gather local preceding assets
        preceding_local = []
        for prev in line_entities[:line_idx]:
            preceding_local.extend(prev)

        # Fallback to rolling global assets if local context lacks antecedents
        all_available_assets = preceding_local if preceding_local else asset_store.get_recent_assets()

        if not all_available_assets:
            resolved_lines.append(resolved_line)
            continue

        primary_asset = all_available_assets[0]
        secondary_asset = all_available_assets[1] if len(all_available_assets) > 1 else primary_asset

        # 1. Resolve plural pronouns ('them', 'these', 'those')
        plural_pattern = re.compile(r'\b(them|these|those)\b', re.IGNORECASE)
        if plural_pattern.search(resolved_line):
            plural_combined = " and ".join(all_available_assets[:2]) if len(all_available_assets) >= 2 else primary_asset
            resolved_line = plural_pattern.sub(plural_combined, resolved_line)

        # 2. Resolve singular pronouns ('this', 'that', 'it')
        action_verbs = ["update", "fix", "check", "handle", "look into", "can you", "review", "renew", "edit", "patch", "sign", "process", "audit", "flush", "restart", "inspect"]
        if any(kw in resolved_line.lower() for kw in action_verbs):
            pronoun_matches = list(re.finditer(r'\b(it|this|that)\b', resolved_line, re.IGNORECASE))
            if pronoun_matches:
                for match in pronoun_matches:
                    p_text = match.group(0).lower()
                    target = secondary_asset if p_text in ["that", "this"] and len(all_available_assets) >= 2 and primary_asset.lower() in resolved_line.lower() else primary_asset
                    resolved_line = re.sub(rf'\b{re.escape(match.group(0))}\b', target, resolved_line, count=1)

        resolved_line = clean_grammar_and_articles(resolved_line)
        resolved_lines.append(resolved_line)

    return resolved_lines


def resolve_coreferences(transcript_text: str) -> str:
    """Primary entry point supporting both short threads and 100-1000 line long threads."""
    lines = [line.strip() for line in transcript_text.strip().split('\n') if line.strip()]
    if not lines:
        return transcript_text

    asset_store = GlobalAssetStore()
    resolved = resolve_coreferences_chunk(lines, asset_store)
    return "\n".join(resolved)