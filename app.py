# --- NEW: Delta System ---
def compute_delta_distribution(numbers_df):
    deltas = []
    for row in numbers_df.values:
        row = sorted(row)
        row_deltas = [row[i+1] - row[i] for i in range(len(row)-1)]
        deltas.extend(row_deltas)
    return Counter(deltas)

def generate_delta_ticket(delta_counter):
    deltas = [d for d,_ in delta_counter.most_common(10)]
    while True:
        start = random.randint(1, 20)  # reasonable start
        ticket = [start]
        for _ in range(5):
            d = random.choice(deltas)
            ticket.append(ticket[-1] + d)
        ticket = [n for n in ticket if 1 <= n <= 49]
        if len(ticket) == 6:
            return sorted(ticket)

# --- NEW: Zone Coverage ---
def generate_zone_ticket(mode="3-zone"):
    if mode == "3-zone":
        low = random.sample(range(1,17), 2)
        mid = random.sample(range(17,34), 2)
        high = random.sample(range(34,50), 2)
        return sorted(low + mid + high)
    else:  # quartiles
        q1 = random.sample(range(1,13), 1)
        q2 = random.sample(range(13,25), 2)
        q3 = random.sample(range(25,37), 2)
        q4 = random.sample(range(37,50), 1)
        return sorted(q1 + q2 + q3 + q4)

# --- NEW: Constraint Filters ---
def passes_constraints(ticket, sum_min, sum_max, spread_min, spread_max, odd_count):
    total = sum(ticket)
    spread = max(ticket) - min(ticket)
    odds = sum(1 for n in ticket if n % 2 == 1)
    evens = 6 - odds
    return (sum_min <= total <= sum_max and
            spread_min <= spread <= spread_max and
            odds == odd_count and evens == 6 - odd_count)

# --- NEW: Smart Exclusion ---
def exclude_numbers(ticket, excluded):
    return [n for n in ticket if n not in excluded]

# --- NEW: Repeat Hit Analysis ---
def compute_repeat_frequency(numbers_df):
    past_draws = [set(row) for row in numbers_df.values.tolist()]
    repeats = Counter()
    for i in range(1, len(past_draws)):
        common = past_draws[i].intersection(past_draws[i-1])
        for n in common:
            repeats[n] += 1
    return repeats

def generate_repeat_ticket(last_draw, repeats, repeat_count=1):
    repeat_nums = random.sample(list(last_draw), repeat_count)
    pool = [n for n in range(1,50) if n not in repeat_nums]
    return sorted(repeat_nums + random.sample(pool, 6 - repeat_count))

# --- NEW: Jackpot Pattern Simulation ---
def simulate_strategy(strategy_func, numbers_df, n=1000):
    past_draws = [set(row) for row in numbers_df.values.tolist()]
    results = {3:0, 4:0, 5:0, 6:0}
    for _ in range(n):
        ticket = set(strategy_func())
        for draw in past_draws:
            hits = len(ticket.intersection(draw))
            if hits >= 3:
                results[hits] += 1
    return results
