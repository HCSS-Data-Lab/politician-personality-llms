from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
DATA_DIR = ROOT_DIR / 'data'
OUTPUTS_DIR = ROOT_DIR / 'outputs'
CACHE_DIR = OUTPUTS_DIR / 'cache'
R: int = 3
REID_R: int = 1
TEMPERATURE: float = 0.7
WORD_CAP: int | None = 2028
OCCASION_TYPES: tuple[str, ...] = ('Campaign', 'Famous', 'Ribbon', 'International')
SCALE_MIN: int = 0
SCALE_MAX: int = 4
_RATINGS_SCHEMA = {'name': 'ratings', 'strict': True, 'schema': {'type': 'object', 'properties': {f's{i}': {'type': 'integer', 'minimum': SCALE_MIN, 'maximum': SCALE_MAX} for i in range(1, 17)}, 'required': [f's{i}' for i in range(1, 17)], 'additionalProperties': False}}
SCALE: tuple[tuple[int, str], ...] = ((0, 'disagree strongly'), (1, 'disagree somewhat'), (2, 'neither disagree nor agree'), (3, 'agree somewhat'), (4, 'agree strongly'))
SCALE_PROMPT_BLOCK: str = '\n'.join((f"{v} '{label}'" for v, label in SCALE))

@dataclass(frozen=True)
class Item:
    id: str
    text: str
    trait: str
    keyed: int
TRAITS_BIG_FIVE: tuple[str, ...] = ('extraversion', 'agreeableness', 'conscientiousness', 'emotional_stability', 'openness')
TRAITS_DARK_TRIAD: tuple[str, ...] = ('narcissism', 'psychopathy', 'machiavellianism')
TRAITS: tuple[str, ...] = TRAITS_BIG_FIVE + TRAITS_DARK_TRIAD
BIG_FIVE_ITEMS: tuple[Item, ...] = (Item('bfi_c1', 'Extraverted, enthusiastic', 'extraversion', +1), Item('bfi_c2', 'Critical, quarrelsome', 'agreeableness', -1), Item('bfi_c3', 'Dependable, self-disciplined', 'conscientiousness', +1), Item('bfi_c4', 'Anxious, easily upset', 'emotional_stability', -1), Item('bfi_c5', 'Open to new experiences, complex', 'openness', +1), Item('bfi_c6', 'Reserved, quiet', 'extraversion', -1), Item('bfi_c7', 'Sympathetic, warm', 'agreeableness', +1), Item('bfi_c8', 'Disorganized, careless', 'conscientiousness', -1), Item('bfi_c9', 'Calm, emotionally stable', 'emotional_stability', +1), Item('bfi_c10', 'Conventional, uncreative', 'openness', -1))
DARK_TRIAD_ITEMS: tuple[Item, ...] = (Item('triad_c1', 'Wants to be admired by others', 'narcissism', +1), Item('triad_c2', 'Shows a lack of remorse', 'psychopathy', +1), Item('triad_c3', 'Might manipulate others to succeed', 'machiavellianism', +1), Item('triad_c4', 'Wants attention from others', 'narcissism', +1), Item('triad_c5', 'Tends to be callous or insensitive', 'psychopathy', +1), Item('triad_c6', 'Tends to use flattery to succeed', 'machiavellianism', +1))
ALL_ITEMS: tuple[Item, ...] = BIG_FIVE_ITEMS + DARK_TRIAD_ITEMS
ITEMS_BY_ID: dict[str, Item] = {it.id: it for it in ALL_ITEMS}
TRAIT_AGG: dict[str, tuple[str, str]] = {'extraversion': ('bfi_c1', 'bfi_c6'), 'agreeableness': ('bfi_c7', 'bfi_c2'), 'conscientiousness': ('bfi_c3', 'bfi_c8'), 'emotional_stability': ('bfi_c9', 'bfi_c4'), 'openness': ('bfi_c5', 'bfi_c10'), 'narcissism': ('triad_c1', 'triad_c4'), 'psychopathy': ('triad_c2', 'triad_c5'), 'machiavellianism': ('triad_c3', 'triad_c6')}
NEGEX_TRAIT_MEAN_VARS: dict[str, str] = {'extraversion': 'bfi_extrav', 'agreeableness': 'bfi_agree', 'conscientiousness': 'bfi_consc', 'emotional_stability': 'bfi_emot', 'openness': 'bfi_open', 'narcissism': 'triad_narciss', 'psychopathy': 'triad_psycho', 'machiavellianism': 'triad_machiav'}
NEGEX_RATER_COUNT_VARS: dict[str, str] = {**{t: 'numrespt_bfi' for t in TRAITS_BIG_FIVE}, **{t: 'numrespt_triad' for t in TRAITS_DARK_TRIAD}}
NEGEX_COMPONENT_SD_PREFIX: str = 'sd'
BIG_FIVE_LEADIN: str = 'Here are a number of personality traits that may or may not apply to {candidate}. Please indicate the extent to which you agree or disagree with the following statements. You should rate the extent to which the pair of traits applies to {candidate}, even if one characteristic applies more strongly than the other.\nIn your opinion, {candidate} might be someone who is...'
DARK_TRIAD_LEADIN: str = 'Next, please indicate the extent to which you agree or disagree with the following statements, related to personality traits that may or may not apply to {candidate}.\nIn your opinion, {candidate} might be someone who ...'
IDENTITY_BLOCK_TEMPLATE: str = '{full_name}, {office} of {country}'
FRAMING_INTRO: str = 'You are taking part in a research study. Below is some information about a public figure. Read it, then rate each of the following descriptive statements according to how much you agree or disagree that it applies to this person, using the scale provided. Base your ratings only on the information given.'
FRAMING_INTRO_KNOWLEDGE: str = 'You are taking part in a research study. Below is the name and role of a public figure. Rate each of the following descriptive statements according to how much you agree or disagree that it applies to this person, using the scale provided. Base your ratings on what you know about this person.'
FRAMING_INTRO_KNOWLEDGE_NUDGE: str = FRAMING_INTRO_KNOWLEDGE + ' Give your best assessment of each statement even if you are not certain.'
REID_PROBE: str = 'Read the following speech excerpt. Who do you think delivered it? Give your single best guess for the speaker, and your confidence from 0 (pure guess) to 100 (certain).'
