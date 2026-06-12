"""Analyze how filler strategy proportions change over training steps.

Reads sample JSONL dumps from filler-tokens-rl runs,
classifies each sample's filler tokens by strategy type, and produces:
1. Terminal tables showing strategy proportions per step (with mean K3 where available)
2. Matplotlib stacked area charts saved as PNGs
3. Per-strategy mean K3 line plots over training steps
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import colorsys

try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    mcolors = None
    plt = None
    mticker = None
import numpy as np

# ---------------------------------------------------------------------------
# Constants imported/replicated from filler_tokens_rl.py
# ---------------------------------------------------------------------------

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]

ANIMALS = [
    "cat", "dog", "elephant", "tiger", "lion", "bear", "wolf", "fox", "rabbit", "deer",
    "horse", "cow", "pig", "sheep", "goat", "chicken", "duck", "goose", "turkey", "eagle",
    "hawk", "owl", "crow", "sparrow", "penguin", "dolphin", "whale", "shark", "salmon", "tuna",
    "octopus", "squid", "crab", "lobster", "shrimp", "snail", "slug", "butterfly", "bee", "ant",
    "spider", "scorpion", "snake", "lizard", "turtle", "frog", "toad", "alligator", "crocodile", "monkey",
    "gorilla", "chimpanzee", "orangutan", "giraffe", "zebra", "hippo", "rhino", "kangaroo", "koala", "panda",
    "raccoon", "skunk", "beaver", "otter", "seal", "walrus", "moose", "elk", "buffalo", "bison",
    "camel", "llama", "alpaca", "donkey", "mule", "parrot", "peacock", "flamingo", "pelican", "seagull",
    "pigeon", "dove", "robin", "cardinal", "bluejay", "woodpecker", "hummingbird", "bat", "mouse", "rat",
    "hamster", "squirrel", "chipmunk", "hedgehog", "porcupine", "armadillo", "sloth", "anteater", "jaguar", "leopard",
]

# Extended animal list for detecting model-invented animal fillers
EXOTIC_ANIMALS = [
    "lemur", "ibex", "jackal", "iguana", "emu", "ibis", "ferret", "heron", "stoat", "gnu",
    "newt", "meerkat", "gerbil", "yak", "quetzal", "tapir", "urial", "dingo", "weasel", "opossum",
    "impala", "jerboa", "gibbon", "jellyfish", "zebu", "egret", "grasshopper", "pangolin", "kinkajou",
    "viper", "vulture", "wombat", "xerus", "numbat", "okapi", "hyena", "cougar", "coyote", "finch",
    "falcon", "gecko", "chameleon", "stork", "crane", "hare", "lynx", "mink", "orca", "pike",
    "quail", "raven", "swan", "tern", "vole", "wren", "adder", "asp", "boa", "condor",
    "dodo", "ermine", "eel", "grouse", "gull", "hornet", "jay", "koi", "lark",
    "loon", "macaw", "magpie", "mantis", "marten", "moth", "narwhal", "nightingale", "osprey", "oyster",
    "pheasant", "puma", "python", "raptor", "rooster", "sable", "starling", "sturgeon", "toucan", "trout",
    "wasp", "wolverine", "bobcat", "caribou", "chinchilla", "cicada", "cockatoo", "coypu", "curlew",
    "dugong", "echidna", "gazelle", "gharial", "grizzly", "grouper", "harrier", "hound",
    "jackrabbit", "lamprey", "loris", "manatee", "marlin", "mongoose", "monitor", "moray",
    "ocelot", "paddlefish", "panther", "peccary", "piranha", "platypus", "polecat", "pronghorn",
    "rattlesnake", "roadrunner", "sailfish", "seahorse", "serval", "skink", "tamarin", "tarsier",
    "tortoise", "vicuna", "wallaby", "warthog", "wildebeest", "wisent", "woodchuck", "zorilla",
]

# Extended fruit list for detecting model-invented fruit fillers
EXOTIC_FRUITS = [
    "quince", "honeydew", "plantain", "soursop", "rambutan", "durian", "tamarind",
    "breadfruit", "mangosteen", "longan", "yuzu", "ackee", "feijoa", "jabuticaba",
    "sapodilla", "cherimoya", "carambola", "langsat", "salak", "pitaya",
    "xigua", "yangmei", "ziziphus", "kiwano", "ximenia", "imbe", "imbu",
    "calamansi", "clementine", "damson", "huckleberry", "jujube", "medlar",
    "pawpaw", "physalis", "pomelo", "quandong", "sloe", "sorb", "ugli",
]

NATO_ALPHABET = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf",
    "Hotel", "India", "Juliet", "Kilo", "Lima", "Mike", "November",
    "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango", "Uniform",
    "Victor", "Whiskey", "X-ray", "Yankee", "Zulu",
]

PI_DIGITS = "31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989"

LOREM_IPSUM = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur Excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum"

RANDOM_TOKENS_VOCAB = {
    "the", "a", "is", "of", "to", "and", "in", "that", "for", "it",
    "as", "on", "with", "be", "at", "by", "this", "from", "or", "an",
    "was", "are", "but", "not", "you", "all", "can", "had", "her", "were",
}

MIXED_FILLER_TYPES = [
    "counting", "fibonacci", "random_numbers", "digit_words", "squares",
    "states", "nato", "colors", "fruits", "lorem", "ellipsis",
    # Legacy types (no longer in mixed filler rotation, but still detected):
    "primes", "pi_digits", "pause", "random_tokens", "alphabet", "animals",
]

COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "cyan", "magenta", "teal", "navy", "maroon",
    "olive", "lime", "coral", "salmon", "ivory", "gold", "silver", "bronze",
    "crimson", "violet", "indigo", "scarlet", "turquoise", "lavender",
    "amber", "emerald", "ruby", "sapphire", "jade", "pearl", "copper",
    "charcoal", "slate", "khaki", "beige", "tan", "mauve", "plum", "peach",
]

DIGIT_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
]

# Pre-compute lookup sets
_US_STATES_LOWER = {s.lower() for s in US_STATES}
_ANIMALS_LOWER = set(ANIMALS)  # already lowercase
_EXOTIC_ANIMALS_LOWER = set(EXOTIC_ANIMALS)  # already lowercase
_ALL_ANIMALS_LOWER = _ANIMALS_LOWER | _EXOTIC_ANIMALS_LOWER
_LOREM_WORDS_LOWER = {w.lower() for w in LOREM_IPSUM.split()}
_NATO_LOWER = {w.lower() for w in NATO_ALPHABET}
_COLORS_LOWER = {c.lower() for c in COLORS}
_DIGIT_WORDS_LOWER = {w.lower() for w in DIGIT_WORDS}

# Pre-compute sequences for matching
def _fibonacci_list(n: int) -> list[str]:
    fibs = [1, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    return [str(x) for x in fibs[:n]]

def _prime_sieve(limit: int) -> list[int]:
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

_FIBONACCI_STRS = _fibonacci_list(500)
_SQUARES_STRS = [str(i * i) for i in range(1, 501)]
_PRIMES = _prime_sieve(50000)
_PRIMES_SET = set(_PRIMES)
_PRIMES_STRS = [str(p) for p in _PRIMES]

# Fruits / cities / shapes for novel pattern detection
_FRUITS = {
    "apple", "banana", "orange", "grape", "mango", "peach", "pear", "plum",
    "cherry", "lemon", "lime", "kiwi", "melon", "watermelon", "strawberry",
    "blueberry", "raspberry", "blackberry", "pineapple", "coconut", "papaya",
    "fig", "date", "apricot", "pomegranate", "guava", "lychee", "tangerine",
    "grapefruit", "avocado", "nectarine", "cantaloupe", "cranberry",
    "dragonfruit", "passionfruit", "starfruit", "persimmon", "kumquat",
    "mulberry", "boysenberry", "gooseberry", "elderberry", "jackfruit",
}

_EXOTIC_FRUITS_LOWER = {f.lower() for f in EXOTIC_FRUITS}
_ALL_FRUITS_LOWER = _FRUITS | _EXOTIC_FRUITS_LOWER

_CITIES = {
    "paris", "london", "berlin", "rome", "madrid", "tokyo", "beijing",
    "moscow", "sydney", "cairo", "mumbai", "bangkok", "istanbul", "vienna",
    "amsterdam", "brussels", "lisbon", "warsaw", "budapest", "stockholm",
    "oslo", "helsinki", "copenhagen", "dublin", "edinburgh", "glasgow",
    "manchester", "birmingham", "liverpool", "bristol", "leeds", "sheffield",
    "nottingham", "plymouth", "southampton", "portsmouth", "brighton", "hove",
}

_SHAPES = {
    "circle", "square", "triangle", "rectangle", "pentagon", "hexagon",
    "octagon", "oval", "diamond", "star", "cube", "sphere", "cylinder",
    "cone", "pyramid", "prism", "trapezoid", "parallelogram", "rhombus",
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_filler(content: str) -> str:
    """Classify the filler strategy used in a sample's content.

    Extracts filler tokens (everything before 'Answer:') and returns
    a string label for the detected strategy.
    """
    # Find "Answer:" boundary
    answer_match = re.search(r'\nAnswer:', content)
    if answer_match is None:
        # Try without newline
        answer_match = re.search(r'Answer:', content)
        if answer_match is None:
            return "invalid"

    filler_text = content[:answer_match.start()].strip()
    if not filler_text:
        return "empty"

    tokens = filler_text.split()
    if not tokens:
        return "empty"

    n = len(tokens)

    # 1. ellipsis — all tokens are "..."
    if all(t == "..." for t in tokens):
        return "ellipsis"

    # 2. pause — all tokens are "pause"
    if all(t.lower() == "pause" for t in tokens):
        return "pause"

    # 3. alphabet — single letters cycling a->z
    if n >= 3 and all(len(t) == 1 and t.isalpha() for t in tokens):
        expected = [chr(ord('a') + i % 26) for i in range(n)]
        if [t.lower() for t in tokens] == expected:
            return "alphabet"

    # 4. pi_digits — single digits matching PI_DIGITS
    if n >= 3 and all(len(t) == 1 and t.isdigit() for t in tokens):
        pi_prefix = list(PI_DIGITS[:n])
        if tokens == pi_prefix:
            return "pi_digits"

    # 5. fibonacci — numbers matching fibonacci sequence
    if n >= 3 and all(t.isdigit() or (t.startswith('-') and t[1:].isdigit()) for t in tokens if t):
        fib_prefix = _FIBONACCI_STRS[:n]
        if tokens == fib_prefix:
            return "fibonacci"

    # 6. primes — numbers matching prime sequence
    if n >= 3:
        primes_prefix = _PRIMES_STRS[:n]
        if tokens == primes_prefix:
            return "primes"

    # 7. counting — sequential integers from 1
    if n >= 3:
        expected_counting = [str(i) for i in range(1, n + 1)]
        if tokens == expected_counting:
            return "counting"

    # 7b. squares — perfect square sequence 1, 4, 9, 16, 25...
    if n >= 3:
        squares_prefix = _SQUARES_STRS[:n]
        if tokens == squares_prefix:
            return "squares"

    # 8. states — tokens are US state names
    # States can be multi-word, so rejoin and try to match
    states_match = _classify_states(filler_text)
    if states_match:
        return "states"

    # 9. nato — tokens match NATO phonetic alphabet words
    if n >= 2 and sum(1 for t in tokens if t.lower() in _NATO_LOWER) / n >= 0.7:
        return "nato"

    # 10. animals — tokens match few-shot ANIMALS list
    if n >= 2 and sum(1 for t in tokens if t.lower() in _ANIMALS_LOWER) / n >= 0.7:
        return "animals"

    # 10b. exotic_animals — tokens match combined animal list but NOT the few-shot list
    if n >= 2 and sum(1 for t in tokens if t.lower() in _ALL_ANIMALS_LOWER) / n >= 0.7:
        return "exotic_animals"

    # 11. lorem — tokens match LOREM_IPSUM words
    if n >= 3 and sum(1 for t in tokens if t.lower() in _LOREM_WORDS_LOWER) / n >= 0.7:
        return "lorem"

    # 11b. colors — tokens match COLORS list
    if n >= 2 and sum(1 for t in tokens if t.lower() in _COLORS_LOWER) / n >= 0.7:
        return "colors"

    # 11c. digit_words — tokens match DIGIT_WORDS list
    if n >= 2 and sum(1 for t in tokens if t.lower() in _DIGIT_WORDS_LOWER) / n >= 0.7:
        return "digit_words"

    # 11d. fruits — tokens match few-shot FRUITS list
    if n >= 2 and sum(1 for t in tokens if t.lower() in _FRUITS) / n >= 0.7:
        return "fruits"

    # 11e. exotic_fruits — tokens match combined fruit list but NOT the few-shot list
    if n >= 2 and sum(1 for t in tokens if t.lower() in _ALL_FRUITS_LOWER) / n >= 0.7:
        return "exotic_fruits"

    # 12. random_tokens — tokens from the 30-word common English vocab
    if n >= 3 and sum(1 for t in tokens if t.lower() in RANDOM_TOKENS_VOCAB) / n >= 0.7:
        return "random_tokens"

    # 13. random_digits — all tokens are single digits (0-9)
    if n >= 3 and all(len(t) == 1 and t.isdigit() for t in tokens):
        return "random_digits"

    # 14. random_numbers — all tokens are digit strings (multi-digit)
    if all(re.match(r'^-?\d+$', t) for t in tokens):
        return "random_numbers"

    # --- Best-effort novel patterns ---

    # reverse_alphabet — single letters z->a
    if n >= 3 and all(len(t) == 1 and t.isalpha() for t in tokens):
        expected_rev = [chr(ord('z') - i % 26) for i in range(n)]
        if [t.lower() for t in tokens] == expected_rev:
            return "reverse_alphabet"

    # even_numbers — even integer sequence 2,4,6,8...
    if n >= 3:
        expected_even = [str(2 * i) for i in range(1, n + 1)]
        if tokens == expected_even:
            return "even_numbers"

    # cities
    lower_text = filler_text.lower()
    city_hits = sum(1 for c in _CITIES if c in lower_text)
    # Rough heuristic: if many city names appear
    if city_hits >= max(3, n * 0.3):
        return "cities"

    # shapes
    if n >= 2 and sum(1 for t in tokens if t.lower() in _SHAPES) / n >= 0.7:
        return "shapes"

    # --- Subcategories of other_alpha ---
    all_alpha = all(t.isalpha() for t in tokens)

    if all_alpha and n >= 3:
        lower_tokens = [t.lower() for t in tokens]

        # repetition — single token dominates (>70%)
        from collections import Counter as _Counter
        most_common_count = _Counter(lower_tokens).most_common(1)[0][1]
        if most_common_count / n >= 0.7:
            return "repetition"

        # exotic_fruits — 30%+ match the combined fruit list (didn't hit 70% above)
        fruit_frac = sum(1 for t in lower_tokens if t in _ALL_FRUITS_LOWER) / n
        if fruit_frac >= 0.3:
            return "exotic_fruits"

        # exotic_animals — 30%+ match the combined animal list (didn't hit 70% above)
        animal_frac = sum(1 for t in lower_tokens if t in _ALL_ANIMALS_LOWER) / n
        if animal_frac >= 0.3:
            return "exotic_animals"

        # mixed — partial matches to multiple known categories
        cat_fracs = [
            sum(1 for t in lower_tokens if t in _NATO_LOWER) / n,
            sum(1 for t in lower_tokens if t in _COLORS_LOWER) / n,
            sum(1 for t in lower_tokens if t in _DIGIT_WORDS_LOWER) / n,
            sum(1 for t in lower_tokens if t in _ALL_FRUITS_LOWER) / n,
            sum(1 for t in lower_tokens if t in _ALL_ANIMALS_LOWER) / n,
        ]
        cats_above_10pct = sum(1 for f in cat_fracs if f >= 0.1)
        if cats_above_10pct >= 2:
            return "mixed"

    # Fallback categories
    if all_alpha:
        return "other_alpha"
    if all(re.match(r'^-?\d+\.?\d*$', t) for t in tokens):
        return "other_numeric"
    return "other"


def _classify_states(text: str) -> bool:
    """Check if text is mostly US state names (handles multi-word states)."""
    remaining = text.strip()
    if not remaining:
        return False

    # Greedily match state names (longest first)
    sorted_states = sorted(US_STATES, key=len, reverse=True)
    lower_remaining = remaining.lower()

    matched_chars = 0
    for state in sorted_states:
        state_lower = state.lower()
        while state_lower in lower_remaining:
            idx = lower_remaining.index(state_lower)
            matched_chars += len(state_lower)
            # Remove the match
            lower_remaining = lower_remaining[:idx] + lower_remaining[idx + len(state_lower):]

    # Check if most of the text was state names
    non_space_total = len(remaining.replace(" ", ""))
    if non_space_total == 0:
        return False
    return matched_chars / non_space_total >= 0.7


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_runs(base_dir: str, min_steps: int = 10) -> list[dict]:
    """Find all runs with enough step files.

    Returns list of dicts with keys: name, samples_dir, step_files
    """
    base = Path(base_dir)
    runs = []

    # Check for samples/ directly under base_dir
    root_samples = base / "samples"
    if root_samples.is_dir():
        step_files = sorted(root_samples.glob("step_*.jsonl"))
        if len(step_files) >= min_steps:
            runs.append({
                "name": "root",
                "samples_dir": root_samples,
                "step_files": step_files,
            })

    # Check subdirectories
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and entry.name != "samples":
            samples_dir = entry / "samples"
            if samples_dir.is_dir():
                step_files = sorted(samples_dir.glob("step_*.jsonl"))
                if len(step_files) >= min_steps:
                    runs.append({
                        "name": entry.name,
                        "samples_dir": samples_dir,
                        "step_files": step_files,
                    })

    return runs


def parse_step_number(path: Path) -> int:
    """Extract step number from filename like step_000012.jsonl."""
    m = re.search(r'step_(\d+)', path.name)
    if m:
        return int(m.group(1))
    return -1


def process_step_file(path: Path) -> tuple[Counter, dict[str, list[float]]]:
    """Read a step JSONL and return counts of each filler strategy and per-strategy K3 values.

    Returns:
        (counts, k3_by_strategy) where k3_by_strategy maps strategy label -> list of K3 values
    """
    counts = Counter()
    k3_by_strategy: dict[str, list[float]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problem = json.loads(line)
            for sample in problem.get("samples", []):
                content = sample.get("content", "")
                label = classify_filler(content)
                counts[label] += 1
                k3_val = sample.get("k3")
                if k3_val is not None:
                    k3_by_strategy[label].append(k3_val)
    return counts, dict(k3_by_strategy)


# ---------------------------------------------------------------------------
# Type ordering helper
# ---------------------------------------------------------------------------

def _get_ordered_types(all_types: set[str]) -> list[str]:
    """Return strategy types in canonical display order."""
    all_types = set(all_types)  # copy to avoid mutating caller
    known_order = MIXED_FILLER_TYPES
    novel_types = ["reverse_alphabet", "even_numbers", "cities", "shapes",
                    "exotic_fruits", "exotic_animals", "repetition", "mixed"]
    fallback_types = ["other_alpha", "other_numeric", "other", "empty", "invalid"]

    ordered = []
    for t in known_order + novel_types + fallback_types:
        if t in all_types:
            ordered.append(t)
            all_types.discard(t)
    ordered.extend(sorted(all_types))
    return ordered


# ---------------------------------------------------------------------------
# Consistent color mapping for strategies across all plots
# ---------------------------------------------------------------------------

_PREFERRED_COLORS: dict[str, str] = {
    "counting":         "#1f77b4",
    "fibonacci":        "#5b9bd5",
    "random_numbers":   "#ff7f0e",
    "states":           "#c55a11",
    "animals":          "#2ca02c",
    "alphabet":         "#17becf",
    "nato":             "#d62728",
    "colors":           "#9467bd",
    "fruits":           "#e74c3c",
    "lorem":            "#98df8a",
    "ellipsis":         "#f1c40f",
    "digit_words":      "#e377c2",
}


def _hex_to_hsv(hex_color: str) -> tuple[float, float, float]:
    """Convert a hex color string to HSV tuple."""
    if mcolors is not None:
        rgb = mcolors.to_rgb(hex_color)
    else:
        stripped = hex_color.lstrip("#")
        rgb = tuple(int(stripped[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(*rgb)


def _generate_dynamic_colors(n: int, exclude_hex: list[str]) -> list[str]:
    """Generate *n* visually distinct hex colors, avoiding *exclude_hex*.

    For n <= 20 we sample from matplotlib's tab20 palette, skipping
    entries whose hue is too close to already-used preferred colors.
    For n > 20 we fall back to golden-ratio hue spacing.
    """
    if n == 0:
        return []

    excluded_hsv = [_hex_to_hsv(h) for h in exclude_hex]

    def _hue_too_close(h: float, threshold: float = 0.08) -> bool:
        for eh, _, _ in excluded_hsv:
            diff = abs(h - eh)
            if min(diff, 1.0 - diff) < threshold:
                return True
        return False

    if n <= 20 and plt is not None and mcolors is not None:
        cmap = plt.get_cmap("tab20")
        candidates = []
        for i in range(20):
            rgba = cmap(i / 20)
            h, s, v = colorsys.rgb_to_hsv(rgba[0], rgba[1], rgba[2])
            if not _hue_too_close(h):
                candidates.append(mcolors.to_hex(rgba[:3]))
            if len(candidates) >= n:
                break
        # If not enough distinct candidates, fill remaining from golden-ratio
        if len(candidates) < n:
            candidates.extend(
                _golden_ratio_colors(n - len(candidates), excluded_hsv)
            )
        return candidates[:n]

    return _golden_ratio_colors(n, excluded_hsv)


def _golden_ratio_colors(n: int, excluded_hsv: list[tuple[float, float, float]]) -> list[str]:
    """Generate *n* colors using golden-ratio hue spacing."""
    golden = 0.618033988749895
    hue = 0.1  # starting hue
    result = []
    attempts = 0
    while len(result) < n and attempts < n * 5:
        r, g, b = colorsys.hsv_to_rgb(hue % 1.0, 0.65, 0.85)
        if mcolors is not None:
            hex_color = mcolors.to_hex((r, g, b))
        else:
            hex_color = "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))
        result.append(hex_color)
        hue += golden
        attempts += 1
    return result


def get_colors_for_strategies(strategies: list[str]) -> list[str]:
    """Return a list of colors for the given strategies, using consistent mapping.

    Strategies present in ``_PREFERRED_COLORS`` keep their fixed color.
    All other strategies are assigned dynamically generated, visually
    distinct colors (sorted alphabetically for determinism).
    """
    preferred_used: dict[str, str] = {}
    dynamic_names: list[str] = []
    for s in strategies:
        if s in _PREFERRED_COLORS:
            preferred_used[s] = _PREFERRED_COLORS[s]
        else:
            dynamic_names.append(s)

    # Deduplicate dynamic names while preserving first-seen order
    seen: set[str] = set()
    unique_dynamic: list[str] = []
    for name in dynamic_names:
        if name not in seen:
            seen.add(name)
            unique_dynamic.append(name)

    # Sort alphabetically for deterministic assignment
    unique_dynamic.sort()

    dynamic_colors = _generate_dynamic_colors(
        len(unique_dynamic), exclude_hex=list(preferred_used.values())
    )
    dynamic_map = dict(zip(unique_dynamic, dynamic_colors))

    return [preferred_used.get(s) or dynamic_map[s] for s in strategies]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(run_name: str, steps: list[int], step_counts: dict[int, Counter],
                step_k3: dict[int, dict[str, list[float]]] | None = None):
    """Print a formatted table of strategy proportions per step, with optional mean K3."""
    # Gather all strategy types that appear
    all_types = set()
    for c in step_counts.values():
        all_types.update(c.keys())

    ordered = _get_ordered_types(all_types)

    # Check if any K3 data exists
    has_k3 = step_k3 is not None and any(bool(v) for v in step_k3.values())

    print(f"\n{'='*80}")
    print(f"Run: {run_name}  ({len(steps)} steps)")
    print(f"{'='*80}")

    # Column widths
    step_col_w = 8
    val_col_w = 7

    # Header
    header = f"{'Step':>{step_col_w}}"
    for t in ordered:
        header += f"  {t:>{max(val_col_w, len(t))}}"
    header += f"  {'Total':>{val_col_w}}"
    if has_k3:
        header += f"  {'meanK3':>{val_col_w}}"
    print(header)
    print("-" * len(header))

    # Rows
    for step in steps:
        counts = step_counts[step]
        total = sum(counts.values())
        row = f"{step:>{step_col_w}}"
        for t in ordered:
            pct = counts.get(t, 0) / total * 100 if total > 0 else 0
            w = max(val_col_w, len(t))
            if pct == 0:
                row += f"  {'·':>{w}}"
            else:
                row += f"  {pct:>{w}.1f}"

        row += f"  {total:>{val_col_w}}"

        if has_k3:
            k3_data = step_k3.get(step, {})
            all_k3_vals = [v for vals in k3_data.values() for v in vals]
            if all_k3_vals:
                mean_k3 = sum(all_k3_vals) / len(all_k3_vals)
                row += f"  {mean_k3:>{val_col_w}.4f}"
            else:
                row += f"  {'—':>{val_col_w}}"

        print(row)

    print()

    # Print per-strategy mean K3 sub-table if K3 data exists
    if has_k3:
        _print_k3_subtable(ordered, steps, step_k3)


def _print_k3_subtable(ordered: list[str], steps: list[int],
                        step_k3: dict[int, dict[str, list[float]]]):
    """Print a sub-table of per-strategy mean K3 values."""
    # Only show strategies that have K3 data somewhere
    strategies_with_k3 = set()
    for k3_data in step_k3.values():
        strategies_with_k3.update(k3_data.keys())

    strats = [t for t in ordered if t in strategies_with_k3]
    if not strats:
        return

    print("  Per-strategy mean K3:")
    step_col_w = 8
    val_col_w = 9

    header = f"  {'Step':>{step_col_w}}"
    for t in strats:
        header += f"  {t:>{max(val_col_w, len(t))}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for step in steps:
        k3_data = step_k3.get(step, {})
        row = f"  {step:>{step_col_w}}"
        for t in strats:
            vals = k3_data.get(t, [])
            w = max(val_col_w, len(t))
            if vals:
                mean_k3 = sum(vals) / len(vals)
                row += f"  {mean_k3:>{w}.4f}"
            else:
                row += f"  {'—':>{w}}"
        print(row)

    print()


def plot_stacked_area(run_name: str, steps: list[int], step_counts: dict[int, Counter], save_path: Path):
    """Create and save a stacked area chart of strategy proportions."""
    if plt is None or mticker is None:
        print("  Skipping stacked area plot: matplotlib is not installed")
        return

    # Gather all types
    all_types = set()
    for c in step_counts.values():
        all_types.update(c.keys())

    ordered = _get_ordered_types(all_types)

    # Build proportion matrix
    x = np.array(steps, dtype=float)
    y = np.zeros((len(ordered), len(steps)))

    for j, step in enumerate(steps):
        counts = step_counts[step]
        total = sum(counts.values())
        if total > 0:
            for i, t in enumerate(ordered):
                y[i, j] = counts.get(t, 0) / total

    # Only show strategies that reach ≥2% at some step in the legend
    LEGEND_THRESHOLD = 0.02
    labels = [t if y[i].max() >= LEGEND_THRESHOLD else "_nolegend_"
              for i, t in enumerate(ordered)]

    colors = get_colors_for_strategies(ordered)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(x, *y, labels=labels, colors=colors, alpha=0.85)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title(f"Filler Strategy Proportions — {run_name}", fontsize=14)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Legend outside the plot (reversed to match bottom-to-top stacking)
    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], leg_labels[::-1],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=True)
    fig.tight_layout()

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {save_path}")


def plot_k3_by_strategy(run_name: str, steps: list[int],
                        step_k3: dict[int, dict[str, list[float]]], save_path: Path):
    """Create and save a line plot of per-strategy mean K3 over training steps."""
    if plt is None or mticker is None:
        print("  Skipping K3 plot: matplotlib is not installed")
        return

    # Find all strategies with K3 data
    strategies_with_k3 = set()
    for k3_data in step_k3.values():
        strategies_with_k3.update(k3_data.keys())

    if not strategies_with_k3:
        return

    ordered = _get_ordered_types(strategies_with_k3)
    ordered = [t for t in ordered if t in strategies_with_k3]

    colors = get_colors_for_strategies(ordered)

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, strategy in enumerate(ordered):
        x_vals = []
        y_vals = []
        for step in steps:
            k3_data = step_k3.get(step, {})
            vals = k3_data.get(strategy, [])
            if vals:
                x_vals.append(step)
                y_vals.append(sum(vals) / len(vals))
        if x_vals:
            ax.plot(x_vals, y_vals, label=strategy, color=colors[i], marker=".", markersize=4, linewidth=1.5)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Mean K3", fontsize=12)
    ax.set_title(f"Per-Strategy Mean K3 — {run_name}", fontsize=14)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=True)
    fig.tight_layout()

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved K3 plot: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "filler-tokens-rl")
    base_dir = os.path.normpath(base_dir)

    if not os.path.isdir(base_dir):
        print(f"Error: directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    runs = find_runs(base_dir)
    if not runs:
        print("No qualifying runs found (need 10+ step files).", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} qualifying run(s) in {base_dir}\n")

    for run in runs:
        name = run["name"]
        step_files = run["step_files"]

        # Parse step numbers and process
        steps = []
        step_counts: dict[int, Counter] = {}
        step_k3: dict[int, dict[str, list[float]]] = {}

        for sf in step_files:
            step_num = parse_step_number(sf)
            if step_num < 0:
                continue
            counts, k3_data = process_step_file(sf)
            steps.append(step_num)
            step_counts[step_num] = counts
            step_k3[step_num] = k3_data

        if not steps:
            continue

        # Print table (with K3 data if available)
        print_table(name, steps, step_counts, step_k3)

        # Generate stacked area plot
        plot_path = run["samples_dir"].parent / "strategy_proportions.png"
        plot_stacked_area(name, steps, step_counts, plot_path)

        # Generate K3 line plot (only if K3 data exists)
        has_k3 = any(bool(v) for v in step_k3.values())
        if has_k3:
            k3_plot_path = run["samples_dir"].parent / "k3_by_strategy.png"
            plot_k3_by_strategy(name, steps, step_k3, k3_plot_path)


if __name__ == "__main__":
    main()
