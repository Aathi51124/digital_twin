import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker()

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# number of records
N = 5000


# =====================================================
# 1 DEVICES DATASET
# =====================================================

devices = []

device_types = [
    ("AC", 1500, 16, 30),
    ("Light", 20, 0, 1),
    ("Fan", 75, 0, 5),
    ("Heater", 2000, 18, 32)
]

rooms = ["LivingRoom", "Bedroom", "Kitchen", "Office"]

for i in range(N):

    dtype = random.choice(device_types)

    devices.append([

        f"D{i}",

        dtype[0] + "_" + str(i),

        random.choice(rooms),

        dtype[0],

        dtype[1],

        random.choice(["ON", "OFF"]),

        random.randint(dtype[2], dtype[3]),

        dtype[2],

        dtype[3]

    ])

devices_df = pd.DataFrame(devices, columns=[

"device_id",
"device_name",
"room",
"type",
"power_rating",
"state",
"value",
"min",
"max"

])

devices_df.to_csv("devices.csv", index=False)


print("devices.csv created")


# =====================================================
# 2 ENVIRONMENT DATASET
# =====================================================

env = []

for i in range(N):

    env.append([

        fake.date_time_this_year(),

        random.choice(rooms),

        random.randint(18, 38),

        random.randint(30, 80),

        random.choice([0,1])

    ])


env_df = pd.DataFrame(env, columns=[

"timestamp",
"room",
"temperature",
"humidity",
"occupancy"

])

env_df.to_csv("environment.csv", index=False)


print("environment.csv created")


# =====================================================
# 3 RULES DATASET
# BALANCED DATASET
# =====================================================

rules = []

for i in range(N):

    condition = random.choice([

        "temperature",
        "occupancy",
        "time"
    ])

    action = random.choice([

        "ON",
        "OFF",
        "SET"
    ])


    priority = random.randint(1,5)

    rules.append([

        f"R{i}",

        condition,

        random.choice(["AC","Fan","Light"]),

        random.choice([">","<","="]),

        random.randint(16,35),

        random.choice(["AC","Fan","Light"]),

        action,

        random.randint(16,30),

        priority

    ])

rules_df = pd.DataFrame(rules, columns=[

"rule_id",
"condition",
"condition_device",
"operator",
"value",
"action_device",
"action",
"action_value",
"priority"

])

rules_df.to_csv("rules.csv", index=False)

print("rules.csv created")


# =====================================================
# 4 ENERGY DATASET
# BALANCED
# =====================================================

energy = []

for i in range(N):

    power = random.uniform(0.01,2.5)

    energy.append([

        fake.date_time_this_year(),

        f"D{i}",

        random.choice(["ON","OFF"]),

        random.randint(16,30),

        round(power,2)

    ])


energy_df = pd.DataFrame(energy, columns=[

"timestamp",
"device_id",
"state",
"value",
"power"

])

energy_df.to_csv("energy.csv", index=False)

print("energy.csv created")


# =====================================================
# 5 CONFLICT DATASET
# BALANCED TARGET
# =====================================================

conflicts = []

for i in range(N):

    conflict = i % 2  # balanced 50%


    conflicts.append([

        random.randint(16,35),

        random.randint(16,35),

        random.randint(1,5),

        random.randint(1,5),

        conflict

    ])

conflict_df = pd.DataFrame(conflicts, columns=[

"rule1_value",
"rule2_value",
"priority1",
"priority2",
"conflict"

])

conflict_df.to_csv("conflict_dataset.csv", index=False)


print("conflict_dataset.csv created")


print("\nALL DATASETS CREATED SUCCESSFULLY")