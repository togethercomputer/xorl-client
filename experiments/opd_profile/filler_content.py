# Vendored bit-identically from /old-data/apanda/tomi/examples/filler_tokens_rl.py
# (the harness that produced the +9.5pp nato / +8.8 random / +7.9 fibonacci numbers on Q3-235B base).
# Provides generate_filler_tokens(n, token_type) -> space-joined string of n elements.
import math, random

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

# NATO phonetic alphabet for filler token generation
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
def fibonacci_sequence(n: int) -> list[int]:
    """Generate first n Fibonacci numbers."""
    fibs = [1, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs[:n]


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
# Pi digits (first 2000 digits after decimal point)
PI_DIGITS = "31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491298336733624406566430860213949463952247371907021798609437027705392171762931767523846748184676694051320005681271452635608277857713427577896091736371787214684409012249534301465495853710507922796892589235420199561121290219608640344181598136297747713099605187072113499999983729780499510597317328160963185950244594553469083026425223082533446850352619311881710100031378387528865875332083814206171776691473035982534904287554687311595628638823537875937519577818577805321712268066130019278766111959092164201989"

# Lorem ipsum text
LOREM_IPSUM = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur Excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum"
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
        import math as _math
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
