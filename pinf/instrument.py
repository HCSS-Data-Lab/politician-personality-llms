# Adapted from the executed study code; see docs/provenance.md.
from __future__ import annotations
import json
import random
import re
SCALE_MIN = 0
SCALE_MAX = 4
TEMPERATURE = 0.7
LABELS: list[str] = [f's{i}' for i in range(1, 17)]
ITEMS: dict[str, str] = {'s1': 'Extraverted, enthusiastic', 's2': 'Critical, quarrelsome', 's3': 'Dependable, self-disciplined', 's4': 'Anxious, easily upset', 's5': 'Open to new experiences, complex', 's6': 'Reserved, quiet', 's7': 'Sympathetic, warm', 's8': 'Disorganized, careless', 's9': 'Calm, emotionally stable', 's10': 'Conventional, uncreative', 's11': 'Wants to be admired by others', 's12': 'Shows a lack of remorse', 's13': 'Might manipulate others to succeed', 's14': 'Wants attention from others', 's15': 'Tends to be callous or insensitive', 's16': 'Tends to use flattery to succeed'}
BIG_FIVE_LABELS: list[str] = [f's{i}' for i in range(1, 11)]
DARK_TRIAD_LABELS: list[str] = [f's{i}' for i in range(11, 17)]
RATINGS_SCHEMA: dict = {'name': 'ratings', 'strict': True, 'schema': {'type': 'object', 'properties': {f's{i}': {'type': 'integer', 'minimum': SCALE_MIN, 'maximum': SCALE_MAX} for i in range(1, 17)}, 'required': [f's{i}' for i in range(1, 17)], 'additionalProperties': False}}
_FRAMING_SCAFFOLD = 'You are taking part in a research study. Below is some information about a public figure. Rate each of the following descriptive statements according to how much you agree or disagree that it applies to this person, using the scale provided.'
FRAMINGS: dict[str, str] = {'G': _FRAMING_SCAFFOLD + ' Base your ratings only on the information given.', 'O': _FRAMING_SCAFFOLD, 'K': _FRAMING_SCAFFOLD + ' Base your ratings on what you know about this person.'}
_ANON_FRAMING_SCAFFOLD = 'You are taking part in a research study. Below is an excerpt from a speech. Rate each of the following descriptive statements according to how much you agree or disagree that it applies to the person who delivered this speech, using the scale provided.'
ANON_FRAMINGS: dict[str, str] = {'G': _ANON_FRAMING_SCAFFOLD + ' Base your ratings only on the information given.', 'O': _ANON_FRAMING_SCAFFOLD}
REID_SCHEMA: dict = {'name': 'reid_guess', 'strict': True, 'schema': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'country': {'type': 'string'}, 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}}, 'required': ['name', 'country', 'confidence'], 'additionalProperties': False}}
_REID_PROMPT = 'Read the following speech excerpt. Identify who most likely delivered it. Respond with a single JSON object and nothing else, with exactly these keys: "name" (your single best guess for the speaker, or "unknown" if you cannot tell), "country" (the country you think they lead or represent, or "unknown"), and "confidence" (a number from 0 for a pure guess to 1 for certain). Do not add any text outside the JSON object.'
_SCALE_BLOCK = "Use the following rating scale, choosing exactly one whole number for each statement:\n0 'disagree strongly'\n1 'disagree somewhat'\n2 'neither disagree nor agree'\n3 'agree somewhat'\n4 'agree strongly'"
_BIG_FIVE_LEADIN = 'Here are a number of personality traits that may or may not apply to this person. Please indicate the extent to which you agree or disagree with the following statements. You should rate the extent to which the pair of traits applies to this person, even if one characteristic applies more strongly than the other.\nIn your opinion, this person might be someone who is...'
_DARK_TRIAD_LEADIN = 'Next, please indicate the extent to which you agree or disagree with the following statements, related to personality traits that may or may not apply to this person.\nIn your opinion, this person might be someone who ...'
_OUTPUT_INSTRUCTION = 'Respond with a single JSON object and nothing else. Include one entry for every statement above: each key is that statement\'s label exactly as shown (for example, "s1"), and each value is a single whole number from 0 to 4. Do not add any text, explanation, or commentary outside the JSON object.'

class ResponseError(Exception):
    pass

class ResponseParseError(ResponseError):
    pass

class ResponseValidationError(ResponseError):
    pass

def item_order(seed: int | None) -> list[str]:
    labels = list(LABELS)
    if seed is None:
        return labels
    random.Random(seed).shuffle(labels)
    return labels

def _render_items(labels: list[str]) -> str:
    return '\n'.join((f'{label}. {ITEMS[label]}' for label in labels))

def _speech_block(speech: str) -> str:
    return f'Speech excerpt:\n---\n{speech}\n---'

def _assemble_blocks(framing_text: str, evidence: str, seed: int | None) -> str:
    order = item_order(seed)
    big_five = [label for label in order if label in BIG_FIVE_LABELS]
    dark_triad = [label for label in order if label in DARK_TRIAD_LABELS]
    blocks = [framing_text, evidence, _SCALE_BLOCK, f'{_BIG_FIVE_LEADIN}\n\n{_render_items(big_five)}', f'{_DARK_TRIAD_LEADIN}\n\n{_render_items(dark_triad)}', _OUTPUT_INSTRUCTION]
    return '\n\n'.join(blocks)

def build_prompt(identity: str, framing: str='G', seed: int | None=None) -> str:
    if framing not in FRAMINGS:
        raise ValueError(f'framing must be one of {sorted(FRAMINGS)}, got {framing!r}')
    return _assemble_blocks(FRAMINGS[framing], f'Public figure: {identity}', seed)

def build_named_speech_prompt(identity: str, speech: str, framing: str='G', seed: int | None=None) -> str:
    if framing not in FRAMINGS:
        raise ValueError(f'framing must be one of {sorted(FRAMINGS)}, got {framing!r}')
    if not speech:
        raise ValueError('named-speech requires a non-empty speech')
    evidence = f'Public figure: {identity}\n\n{_speech_block(speech)}'
    return _assemble_blocks(FRAMINGS[framing], evidence, seed)

def build_anon_speech_prompt(speech: str, framing: str='G', seed: int | None=None) -> str:
    if framing not in ANON_FRAMINGS:
        raise ValueError(f'anon framing must be one of {sorted(ANON_FRAMINGS)}, got {framing!r}')
    if not speech:
        raise ValueError('anon-speech requires a non-empty speech')
    return _assemble_blocks(ANON_FRAMINGS[framing], _speech_block(speech), seed)

def build_reid_prompt(speech: str) -> str:
    if not speech:
        raise ValueError('build_reid_prompt requires a non-empty speech')
    return f'{_REID_PROMPT}\n\n{_speech_block(speech)}'

def reid_extra_body(hf_id: str | None=None) -> dict:
    body: dict = {'response_format': {'type': 'json_schema', 'json_schema': REID_SCHEMA}}
    if hf_id is None or 'qwen' in hf_id.lower():
        body['chat_template_kwargs'] = {'enable_thinking': False}
    return body

def parse_reid_response(text: str) -> dict:
    obj = _extract_object(text or '')
    if obj is None:
        raise ResponseParseError(f'no JSON object found in re-id response: {text[:120]!r}')
    try:
        name = str(obj['name']).strip()
        country = str(obj['country']).strip()
        confidence = float(obj['confidence'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ResponseValidationError(f're-id JSON missing/ill-typed fields: {obj!r} ({exc})')
    if confidence > 1.0:
        confidence = confidence / 100.0 if confidence <= 100.0 else 1.0
    confidence = max(0.0, min(1.0, confidence))
    return {'name': name, 'country': country, 'confidence': confidence}

def vllm_extra_body(hf_id: str | None=None) -> dict:
    body: dict = {'response_format': {'type': 'json_schema', 'json_schema': RATINGS_SCHEMA}}
    if hf_id is None or 'qwen' in hf_id.lower():
        body['chat_template_kwargs'] = {'enable_thinking': False}
    return body
_FENCE_RE = re.compile('```(?:json)?\\s*(.*?)```', re.DOTALL | re.IGNORECASE)
_SLABEL_RE = re.compile('^s0*([1-9]\\d*)$')

def _try_load(snippet: str) -> dict | None:
    try:
        obj = json.loads(snippet)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None

def _extract_object(text: str) -> dict | None:
    for m in _FENCE_RE.finditer(text):
        obj = _try_load(m.group(1).strip())
        if obj is not None:
            return obj
    obj = _try_load(text.strip())
    if obj is not None:
        return obj
    depth, start = (0, None)
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}' and depth:
            depth -= 1
            if depth == 0 and start is not None:
                obj = _try_load(text[start:i + 1])
                if obj is not None:
                    return obj
    return None

def _normalize_label(raw: str) -> str | None:
    m = _SLABEL_RE.match(str(raw).strip())
    if not m:
        return None
    n = int(m.group(1))
    return f's{n}' if 1 <= n <= 16 else None

def _coerce_rating(value, label: str) -> int:
    if isinstance(value, bool):
        raise ResponseValidationError(f'{label}: boolean is not a valid rating')
    if isinstance(value, int):
        iv = value
    elif isinstance(value, float) and value.is_integer():
        iv = int(value)
    elif isinstance(value, str):
        try:
            iv = int(value.strip())
        except ValueError:
            raise ResponseValidationError(f'{label}: {value!r} is not an integer')
    else:
        raise ResponseValidationError(f'{label}: {value!r} is not an integer')
    if not SCALE_MIN <= iv <= SCALE_MAX:
        raise ResponseValidationError(f'{label}: {iv} out of range [{SCALE_MIN},{SCALE_MAX}]')
    return iv

def parse_response(text: str) -> dict[str, int]:
    obj = _extract_object(text or '')
    if obj is None:
        raise ResponseParseError(f'no JSON object found in response: {text[:120]!r}')
    ratings: dict[str, int] = {}
    for raw_key, raw_val in obj.items():
        label = _normalize_label(raw_key)
        if label is None:
            continue
        ratings[label] = _coerce_rating(raw_val, label)
    missing = [label for label in LABELS if label not in ratings]
    if missing:
        raise ResponseValidationError(f'missing ratings for {missing}')
    return {label: ratings[label] for label in LABELS}
