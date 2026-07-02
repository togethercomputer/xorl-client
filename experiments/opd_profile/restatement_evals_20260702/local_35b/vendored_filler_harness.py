"""Vendored VERBATIM (ast-extracted, 2026-07-02) from
/home/apanda/xorl-client-chat-completions/examples/filler_tokens_rl.py
so the step-0 restatement evals use the EXACT training-harness problem generators,
system prompts, and filler builders without importing the harness's chz/tinker deps.
Do not edit the extracted bodies."""
import random
import math


PROBLEM_DESCRIPTIONS = {
    "arithmetic": "You will be given a math problem.",
    "multiplication_3digit": "You will be given a 3-digit multiplication problem.",
    "multiplication_4digit": "You will be given a 4-digit multiplication problem.",
    "multiplication_5digit": "You will be given a 5-digit multiplication problem.",
    "multiplication_6digit": "You will be given a 6-digit multiplication problem.",
    "multiplication_4_3": "You will be given a multiplication problem (4-digit × 3-digit).",
    "count_reassignments": "You will be given a short code snippet and must count how many times a variable is reassigned.",
}


FILLER_INSTRUCTIONS = {
    "counting": 'Before answering, generate up to {filler_tokens} filler tokens (numbers counting from 1 to N, e.g. "1 2 3 4 5...") to give yourself time to process the problem. You may only generate up to {filler_tokens} tokens. Note that each digit you generate is a separate token. Format:\n[filler tokens]\nAnswer: [number]',
    "mixed": (
        'Do not reason explicitly or show your work. Instead, you may generate up to {filler_tokens} filler tokens '
        'to help you process the problem internally. You may reason in an encoded manner through your choice and ordering of these filler tokens. These can be any of the following sequences: '
        'counting numbers ("1 2 3 4 5..."), Fibonacci numbers ("1 1 2 3 5 8 13..."), '
        'random numbers, number words ("one two three..."), square numbers ("1 4 9 16 25..."), '
        'US state names ("Texas Maine Ohio Alaska..."), '
        'NATO phonetic alphabet words ("Golf Echo Bravo Lima..."), '
        'color names ("coral emerald navy teal..."), fruit names ("mango kiwi plum cherry..."), '
        'lorem ipsum words, ellipses ("... ... ..."), or similar sequences.\n'
        'You may only generate up to {filler_tokens} tokens. Note that each digit you generate is a separate token.\n'
        'Format:\n[filler tokens]\nAnswer: [number]'
    ),
    "mixed_opus_pref": (
        'Do not reason explicitly or show your work. Instead, you may generate up to {filler_tokens} filler tokens '
        'to help you process the problem internally. You may reason in an encoded manner through your choice and ordering of these filler tokens. These can be any of the following sequences: '
        'counting numbers ("1 2 3 4 5..."), Fibonacci numbers ("1 1 2 3 5 8 13..."), '
        'random numbers, random single digits ("1 9 3 8 7 6 0 2..."), random common words ("the was for it a on is of..."), '
        'square numbers ("1 4 9 16 25..."), prime numbers ("2 3 5 7 11 13..."), '
        'digits of pi ("3 1 4 1 5 9 2 6..."), '
        'ellipses ("... ... ..."), the word "pause" repeated ("pause pause pause..."), lorem ipsum words, or similar sequences.\n'
        'You may only generate up to {filler_tokens} tokens. Note that each digit you generate is a separate token. '
        'Each "..." is one token. Each "pause" is one token.\n'
        'Format:\n[filler tokens]\nAnswer: [number]'
    ),
    "animal": 'Before answering, generate the name of an animal (e.g. "elephant", "tiger", "dolphin") to give yourself extra processing time. Format:\n[animal]\nAnswer: [number]',
    "v3": 'Format:\nAnswer: [number]',
    "random_tokens": (
        'Do not reason explicitly or show your work. Instead, you may generate up to {filler_tokens} filler tokens '
        'to help you process the problem internally. You may reason in an encoded manner through your choice and ordering of these filler tokens. '
        'These should be random common words (e.g. "the was for it a on is of to and by at be this or with from not all can").\n'
        'You may only generate up to {filler_tokens} tokens.\n'
        'Format:\n[filler tokens]\nAnswer: [number]'
    ),
    "random_numbers": (
        'Do not reason explicitly or show your work. Instead, you may generate up to {filler_tokens} filler tokens '
        'to help you process the problem internally. You may reason in an encoded manner through your choice and ordering of these filler tokens. '
        'These should be random numbers (e.g. "482 17 935 6 204 73...").\n'
        'You may only generate up to {filler_tokens} tokens. Note that each digit you generate is a separate token.\n'
        'Format:\n[filler tokens]\nAnswer: [number]'
    ),
}


ANSWER_FORMAT = "Answer immediately using the format 'Answer: [ANSWER]' where [ANSWER] is just the numerical answer, nothing else. No explanation, no words, no reasoning, just the number."


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


NATO_ALPHABET = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf",
    "Hotel", "India", "Juliet", "Kilo", "Lima", "Mike", "November",
    "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango", "Uniform",
    "Victor", "Whiskey", "X-ray", "Yankee", "Zulu",
]


COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "cyan", "magenta", "teal", "navy", "maroon",
    "olive", "lime", "coral", "salmon", "ivory", "gold", "silver", "bronze",
    "crimson", "violet", "indigo", "scarlet", "turquoise", "lavender",
    "amber", "emerald", "ruby", "sapphire", "jade", "pearl", "copper",
    "charcoal", "slate", "khaki", "beige", "tan", "mauve", "plum", "peach",
]


FRUITS = [
    "apple", "banana", "orange", "grape", "mango", "peach", "pear", "plum",
    "cherry", "lemon", "lime", "kiwi", "melon", "watermelon", "strawberry",
    "blueberry", "raspberry", "blackberry", "pineapple", "coconut", "papaya",
    "fig", "date", "apricot", "pomegranate", "guava", "lychee", "tangerine",
    "grapefruit", "avocado", "nectarine", "cantaloupe", "cranberry",
    "dragonfruit", "passionfruit", "starfruit", "persimmon", "kumquat",
    "mulberry", "boysenberry", "gooseberry", "elderberry", "jackfruit",
]


DIGIT_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
]


MIXED_FILLER_TYPES = [
    "counting",
    "fibonacci",
    "random_numbers",
    "digit_words",
    "squares",
    "states",
    "nato",
    "colors",
    "fruits",
    "lorem",
    "ellipsis",
]


MIXED_FILLER_TYPES_OPUS_PREF = [
    "counting",
    "fibonacci",
    "random_numbers",
    "random_digits",
    "random_tokens",
    "squares",
    "primes",
    "pi_digits",
    "ellipsis",
    "pause",
    "lorem",
]


def get_system_prompt(problem_type: str, mixed_filler: bool = False, filler_token_type: str = "counting", filler_tokens: int = 100, opus_pref_filler: bool = False) -> str:
    """Get the system prompt for a given problem type and filler type.

    Composes the prompt from independent problem_type and filler_token_type.

    Args:
        problem_type: Type of math problem (arithmetic, multiplication_3digit, etc.)
        mixed_filler: If True, use mixed filler instruction
        filler_token_type: Type of filler tokens (counting, animal, etc.)
        filler_tokens: Number of filler tokens to suggest in the prompt (default 100).
                      If 0, no filler instruction is included (direct answer mode).
    """
    if problem_type not in PROBLEM_DESCRIPTIONS:
        raise ValueError(f"Unknown problem_type: {problem_type}. Valid types: {list(PROBLEM_DESCRIPTIONS.keys())}")

    problem_desc = PROBLEM_DESCRIPTIONS[problem_type]

    # V3 mode: simple format instruction, no filler tokens needed
    if filler_token_type == "v3":
        filler_instr = FILLER_INSTRUCTIONS["v3"]
        return f"/no_think\n{problem_desc}\n\n{filler_instr}"

    # No filler mode: just problem description + answer format
    if filler_tokens == 0:
        return f"/no_think\n{problem_desc} {ANSWER_FORMAT}"

    # Determine filler type
    if mixed_filler:
        filler_key = "mixed_opus_pref" if opus_pref_filler else "mixed"
    elif filler_token_type == "animal":
        filler_key = "animal"
    elif filler_token_type in FILLER_INSTRUCTIONS:
        filler_key = filler_token_type
    else:
        filler_key = "counting"

    # Compose the prompt with filler instruction
    filler_instr = FILLER_INSTRUCTIONS[filler_key].format(filler_tokens=filler_tokens)

    return f"/no_think\n{problem_desc}\n\n{filler_instr}"


def generate_multiplication_problems(n: int, num_digits: int = 3, seed: int = 12345) -> list[dict]:
    """Generate n multiplication problems with specified number of digits.

    Each problem is a dict with 'problem' and 'answer' keys.
    Problems are of the form "What is XXXX * YYYY?" where XXXX and YYYY have num_digits digits.
    """
    rng = random.Random(seed)
    problems = []
    min_val = 10 ** (num_digits - 1)  # e.g., 100 for 3 digits, 1000 for 4 digits
    max_val = 10 ** num_digits - 1    # e.g., 999 for 3 digits, 9999 for 4 digits
    for _ in range(n):
        a = rng.randint(min_val, max_val)
        b = rng.randint(min_val, max_val)
        problem = f"What is {a} * {b}?"
        answer = a * b
        problems.append({"problem": problem, "answer": answer})
    return problems


_CODE_VARS = [
    "x", "y", "z", "a", "b", "c", "n", "m", "k", "p", "q", "r", "s", "t",
    "count", "total", "result", "val", "tmp", "acc", "idx", "cur", "prev", "best",
]


def generate_variable_reassignment_problems(
    n: int,
    seed: int = 12345,
    # Harder default calibration (2026-07-02): higher reassignment counts + many more distractor
    # lines push per-problem accuracy down into the contested band (~0.4-0.6) so GRPO groups are
    # naturally mixed (low ZA) with headroom before saturation. Tune these if step-0 accuracy is
    # off-band.
    min_reassign: int = 5,
    max_reassign: int = 13,
    min_distractors: int = 22,
    max_distractors: int = 42,
) -> list[dict]:
    """Generate n code-comprehension problems: 'how many times is variable X reassigned?'.

    A scan-and-count task (not arithmetic). Each snippet has a target variable assigned
    K = reassignments+1 times (the first is its initial definition; the rest are reassignments,
    which is the answer) interleaved with DISTRACTOR statements: assignments to OTHER variables,
    some of which *read* the target on the RHS (a use, which must NOT be counted). The first line
    is always the target's initial assignment (define-before-use). Difficulty scales with the
    reassignment count and the number of distractor lines. Answer = reassignments (a small int).
    """
    rng = random.Random(seed)
    problems = []
    for _ in range(n):
        nvars = rng.randint(3, 5)
        vars = rng.sample(_CODE_VARS, nvars)
        target = vars[0]
        others = vars[1:]
        reassignments = rng.randint(min_reassign, max_reassign)
        total_target_assigns = reassignments + 1
        n_distract = rng.randint(min_distractors, max_distractors)

        def _target_assign():
            if others and rng.random() < 0.5:
                o = rng.choice(others); op = rng.choice(["+", "-", "*"])
                return f"{target} = {o} {op} {rng.randint(1, 9)}"
            return f"{target} = {rng.randint(0, 99)}"

        def _distractor():
            o = rng.choice(others) if others else target
            if others and rng.random() < 0.5:
                # RHS *reads* the target (a use, NOT a reassignment of target)
                op = rng.choice(["+", "-", "*"])
                return f"{o} = {target} {op} {rng.randint(1, 9)}"
            src = rng.choice(vars)
            return f"{o} = {src} + {rng.randint(0, 9)}"

        # First line = target's initial assignment; remaining target assigns + distractors shuffled.
        rest = [_target_assign() for _ in range(reassignments)] + [_distractor() for _ in range(n_distract)]
        rng.shuffle(rest)
        code = "\n".join([_target_assign()] + rest)
        problem = (
            f"In the following code, how many times is the variable `{target}` reassigned "
            f"(assigned a new value after its first assignment)? Reply with just the number.\n\n"
            f"```python\n{code}\n```"
        )
        problems.append({"problem": problem, "answer": reassignments})
    return problems


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


def get_n_primes(n: int) -> list[int]:
    """Get first n prime numbers."""
    # Estimate upper bound for nth prime
    if n < 6:
        limit = 15
    else:
        limit = int(n * (math.log(n) + math.log(math.log(n)))) + 100
    primes = prime_sieve(limit)
    while len(primes) < n:
        limit *= 2
        primes = prime_sieve(limit)
    return primes[:n]


PI_DIGITS = "31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989"


LOREM_IPSUM = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur Excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum"


CITIES = [
    "Tokyo", "Delhi", "Shanghai", "London", "Paris", "Cairo", "Mumbai", "Beijing",
    "Moscow", "Istanbul", "Lagos", "Karachi", "Berlin", "Madrid", "Toronto", "Sydney",
    "Chicago", "Boston", "Seattle", "Denver", "Austin", "Dallas", "Miami", "Atlanta",
    "Nairobi", "Lima", "Bogota", "Santiago", "Jakarta", "Manila", "Bangkok", "Hanoi",
    "Seoul", "Osaka", "Toronto", "Montreal", "Vienna", "Prague", "Warsaw", "Athens",
]


def _element_generator(token_type: str):
    """Yield elements indefinitely for a given filler type."""
    if token_type == "counting":
        i = 1
        while True:
            yield str(i)
            i += 1

    elif token_type == "fibonacci":
        a, b = 1, 1
        while True:
            yield str(a)
            a, b = b, a + b

    elif token_type == "states":
        shuffled = list(US_STATES)
        random.shuffle(shuffled)
        idx = 0
        while True:
            yield shuffled[idx % len(shuffled)]
            idx += 1
            if idx % len(shuffled) == 0:
                random.shuffle(shuffled)

    elif token_type == "primes":
        # Yield primes indefinitely using a simple sieve approach
        primes = get_n_primes(1000)  # Pre-generate a batch
        idx = 0
        while True:
            if idx >= len(primes):
                # Generate more primes if needed
                primes = get_n_primes(len(primes) * 2)
            yield str(primes[idx])
            idx += 1

    elif token_type == "pi_digits":
        idx = 0
        while True:
            yield PI_DIGITS[idx % len(PI_DIGITS)]
            idx += 1

    elif token_type == "ellipsis":
        while True:
            yield "..."

    elif token_type == "lorem":
        words = LOREM_IPSUM.split()
        while True:
            yield random.choice(words)

    elif token_type == "pause":
        while True:
            yield "pause"

    elif token_type == "cities":
        shuffled = list(CITIES)
        random.shuffle(shuffled)
        idx = 0
        while True:
            yield shuffled[idx % len(shuffled)]
            idx += 1
            if idx % len(shuffled) == 0:
                random.shuffle(shuffled)

    elif token_type == "random_numbers":
        while True:
            yield str(random.randint(0, 999))

    elif token_type == "random_tokens":
        vocab = ["the", "a", "is", "of", "to", "and", "in", "that", "for", "it",
                 "as", "on", "with", "be", "at", "by", "this", "from", "or", "an",
                 "was", "are", "but", "not", "you", "all", "can", "had", "her", "were"]
        while True:
            yield random.choice(vocab)

    elif token_type == "alphabet":
        idx = 0
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        while True:
            yield alphabet[idx % 26]
            idx += 1

    elif token_type == "animals":
        while True:
            yield random.choice(ANIMALS)

    elif token_type == "nato":
        while True:
            yield random.choice(NATO_ALPHABET)

    elif token_type == "colors":
        while True:
            yield random.choice(COLORS)

    elif token_type == "fruits":
        while True:
            yield random.choice(FRUITS)

    elif token_type == "digit_words":
        while True:
            yield random.choice(DIGIT_WORDS)

    elif token_type == "fibonacci_shuffled":
        # Same fibonacci numbers but in random order
        a, b = 1, 1
        fib_nums = []
        for _ in range(200):
            fib_nums.append(str(a))
            a, b = b, a + b
        random.shuffle(fib_nums)
        idx = 0
        while True:
            yield fib_nums[idx % len(fib_nums)]
            idx += 1

    elif token_type == "fibonacci_repeated_1":
        # Just "1" repeated (first fibonacci number)
        while True:
            yield "1"

    elif token_type == "fibonacci_repeated_8":
        # Just "8" repeated (6th fibonacci number)
        while True:
            yield "8"

    elif token_type == "fibonacci_repeated_55":
        # Just "55" repeated (10th fibonacci number)
        while True:
            yield "55"

    elif token_type == "random_numbers_magnitude_matched":
        # Random numbers with same magnitude distribution as fibonacci
        # Fibonacci 1-100 tokens: range from 1 to ~3.5e20
        # Sample uniformly in log space
        while True:
            log_val = random.uniform(0, 20)
            yield str(int(10 ** log_val))

    elif token_type == "counting_shuffled":
        # 1-200 in random order
        nums = [str(i) for i in range(1, 201)]
        random.shuffle(nums)
        idx = 0
        while True:
            yield nums[idx % len(nums)]
            idx += 1

    elif token_type.startswith("mixed_fibonacci_animals_"):
        # Mixed filler: X% fibonacci + (100-X)% animals
        pct = int(token_type.split("_")[-1])
        fib_gen = _element_generator("fibonacci")
        while True:
            if random.randint(1, 100) <= pct:
                yield next(fib_gen)
            else:
                yield random.choice(ANIMALS)

    elif token_type == "random_digits":
        while True:
            yield str(random.randint(0, 9))

    elif token_type == "squares":
        i = 1
        while True:
            yield str(i * i)
            i += 1

    else:
        raise ValueError(f"Unknown filler_token_type: {token_type}")


def generate_filler_tokens(n: int, token_type: str = "counting", tokenizer=None) -> str:
    """Generate filler tokens based on type.

    Args:
        n: Target number of tokens to generate (if tokenizer provided) or elements (if not)
        token_type: One of: counting, fibonacci, states, primes, pi_digits, ellipsis, lorem, pause, random_numbers, random_tokens, animals, nato, alphabet
        tokenizer: If provided, generates approximately n tokens. Otherwise generates n elements.
    """
    # Special case: single animal (non-mixed mode)
    if token_type == "animal":
        return random.choice(ANIMALS)

    if tokenizer is None:
        # Fall back to element-based generation (original behavior)
        gen = _element_generator(token_type)
        elements = [next(gen) for _ in range(n)]
        return " ".join(elements)

    # Token-based generation: generate elements until we hit n tokens
    gen = _element_generator(token_type)
    elements = []

    for element in gen:
        elements.append(element)
        # Check token count of current string
        test_str = " ".join(elements)
        token_count = len(tokenizer.encode(test_str))

        if token_count >= n:
            break

    return " ".join(elements)


MEGA_FILLER_TYPES = [
    "random_numbers", "counting", "fibonacci", "primes", "squares",
    "colors", "fruits", "states", "cities", "lorem", "pause", "ellipsis",
]


def generate_mega_filler(target_tokens: int, tokenizer=None) -> str:
    """Concatenate ALL sequence types back-to-back into a ~target_tokens filler blob.

    Each type contributes ~target_tokens/len(types) tokens; the type order is shuffled per
    call so the blob differs across problems. Within a GRPO group the filler is fixed (built
    once per problem) and answer sampling provides the variation.
    """
    if target_tokens <= 0:
        # Condition C (question-restated, NO filler): mega_filler_tokens=0 -> empty blob so the
        # prescribe prefill is just "\n{question}\nAnswer:" (isolates the question-restatement from
        # the filler-compute effect). Without this guard `per=max(1,...)` floors at ~1 tok/type.
        return ""
    types = list(MEGA_FILLER_TYPES)
    random.shuffle(types)
    per = max(1, target_tokens // len(types))
    chunks = [generate_filler_tokens(per, t, tokenizer) for t in types]
    blob = " ".join(c for c in chunks if c)
    if tokenizer is not None:
        # top up to the target (defensive; constant types like pause/ellipsis undershoot)
        guard = 0
        while len(tokenizer.encode(blob)) < target_tokens and guard < 64:
            blob = blob + " " + generate_filler_tokens(per, random.choice(types), tokenizer)
            guard += 1
    return blob


def prime_sieve(limit: int) -> list[int]:
    """Generate primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]
