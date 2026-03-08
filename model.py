import pandas as pd
import numpy as np
import random
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import pulp

# Download NLTK data (uncomment if not present)
# nltk.download('punkt')
# nltk.download('stopwords')

# =============================================================================
# PART 1: DIGITAL TWIN SIMULATOR
# =============================================================================
class DigitalTwinSimulator:
    def __init__(self, devices_df, environment_df, rules_df):
        self.devices = devices_df.to_dict('records')
        self.environment = environment_df.to_dict('records')
        self.rules = rules_df.to_dict('records')
        self.virtual_home = []
        self.total_energy = 0.0
        self.comfort_score = 0
        self.time_step_results = []

    def initialize_virtual_home(self):
        for d in self.devices:
            self.virtual_home.append({
                'id': d['device_id'],
                'state': d['state'],
                'power': d['power_rating'],
                'current_state': d['state']
            })

    def apply_rule(self, rule, env):
        device_id = rule['device_id']
        condition_field = rule['condition_field']
        operator = rule['operator']
        threshold = rule['threshold']
        action = rule['action']

        # Get current environmental value
        value = env.get(condition_field)
        if value is None:
            return

        # Evaluate condition
        condition_met = False
        if operator == '>':
            condition_met = value > threshold
        elif operator == '<':
            condition_met = value < threshold
        elif operator == '==':
            condition_met = value == threshold
        elif operator == '>=':
            condition_met = value >= threshold
        elif operator == '<=':
            condition_met = value <= threshold

        if condition_met:
            for dev in self.virtual_home:
                if dev['id'] == device_id:
                    dev['current_state'] = action
                    break

    def calculate_energy(self, time_interval):
        energy_step = 0.0
        for dev in self.virtual_home:
            if dev['current_state'] == 'ON':
                energy_step += dev['power'] * time_interval
        return energy_step

    def calculate_comfort(self, temperature):
        return 1 if 22 <= temperature <= 26 else -1

    def run_simulation(self, time_interval=1.0):
        self.initialize_virtual_home()
        for t, env in enumerate(self.environment):
            # Apply all rules (order may matter; here we apply sequentially)
            for rule in self.rules:
                self.apply_rule(rule, env)

            energy_step = self.calculate_energy(time_interval)
            self.total_energy += energy_step

            comfort_step = self.calculate_comfort(env['temperature'])
            self.comfort_score += comfort_step

            self.time_step_results.append({
                'time_step': t,
                'temperature': env['temperature'],
                'humidity': env['humidity'],
                'occupancy': env['occupancy'],
                'device_states': [dev['current_state'] for dev in self.virtual_home],
                'energy_step': energy_step,
                'cumulative_energy': self.total_energy,
                'cumulative_comfort': self.comfort_score
            })
        return pd.DataFrame(self.time_step_results)


# =============================================================================
# PART 2: CONFLICT DETECTION USING RULE DEPENDENCY GRAPH
# =============================================================================
class ConflictDetector:
    def __init__(self, rules_df):
        self.rules = rules_df.to_dict('records')
        self.graph = defaultdict(list)

    def build_graph(self):
        for i, rule in enumerate(self.rules):
            self.graph[rule['device_id']].append(i)

    def detect_conflicts(self):
        conflicts = []
        for device, indices in self.graph.items():
            if len(indices) > 1:
                # Compare each pair for contradiction
                for idx in indices:
                    for jdx in indices:
                        if idx >= jdx:
                            continue
                        r1 = self.rules[idx]
                        r2 = self.rules[jdx]
                        # Contradiction: same condition field, different actions
                        if (r1['condition_field'] == r2['condition_field'] and
                            r1['action'] != r2['action']):
                            conflicts.append((idx, jdx))
        return conflicts

    def resolve_conflict(self, conflict_pair):
        idx, jdx = conflict_pair
        # Higher priority = lower number
        if self.rules[idx]['priority'] < self.rules[jdx]['priority']:
            return self.rules[idx]
        else:
            return self.rules[jdx]

    def run(self):
        self.build_graph()
        conflicts = self.detect_conflicts()
        resolved_rules = []
        for conflict in conflicts:
            resolved_rules.append(self.resolve_conflict(conflict))
        return resolved_rules


# =============================================================================
# PART 3: LSTM ENERGY PREDICTION
# =============================================================================
class LSTMEnergyPredictor:
    def __init__(self, energy_df, sequence_length=10):
        self.energy = energy_df['energy'].values.astype(float)
        self.seq_len = sequence_length
        self.scaler_mean = None
        self.scaler_std = None
        self.model = None

    def preprocess(self):
        # Normalize
        self.scaler_mean = self.energy.mean()
        self.scaler_std = self.energy.std()
        norm_vals = (self.energy - self.scaler_mean) / self.scaler_std

        # Create sequences
        X, y = [], []
        for i in range(len(norm_vals) - self.seq_len):
            X.append(norm_vals[i:i+self.seq_len])
            y.append(norm_vals[i+self.seq_len])
        X = np.array(X).reshape(-1, self.seq_len, 1)
        y = np.array(y)
        return X, y

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=input_shape))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, epochs=50, validation_split=0.2):
        X, y = self.preprocess()
        # Split manually to keep track
        split = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model = self.build_model((self.seq_len, 1))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                       epochs=epochs, batch_size=32, callbacks=[early_stop], verbose=0)

        # Evaluate
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(np.mean((y_pred.flatten() - y_test)**2))
        # Denormalize RMSE for interpretability
        rmse_denorm = rmse * self.scaler_std
        print(f"LSTM Energy Prediction RMSE (normalized): {rmse:.4f}, denormalized: {rmse_denorm:.2f}")
        return rmse_denorm

    def predict_next(self, last_sequence):
        # last_sequence: list of last 'seq_len' raw values
        last_norm = (np.array(last_sequence) - self.scaler_mean) / self.scaler_std
        last_norm = last_norm.reshape(1, self.seq_len, 1)
        pred_norm = self.model.predict(last_norm, verbose=0)[0,0]
        pred = pred_norm * self.scaler_std + self.scaler_mean
        return pred


# =============================================================================
# PART 4: LINEAR PROGRAMMING OPTIMIZATION
# =============================================================================
class LPOptimizer:
    """
    Optimizes device settings to minimize energy while maintaining comfort.
    Uses a simple thermal model:
        temp_next = temp_current + 2*heater_on - 2*cooler_on - 0.1*(temp_current - 20)
    Constraints:
        - If occupancy=1, lights must be on.
        - Temperature after one time step must be between 22 and 26.
    Decision variables: binary for each device.
    """
    def __init__(self, devices_df, current_temp, occupancy, time_interval=1.0):
        self.devices = devices_df.to_dict('records')
        self.current_temp = current_temp
        self.occupancy = occupancy
        self.dt = time_interval
        self.outside_temp = 20.0  # assumed constant

    def optimize(self):
        # Create a LP problem
        prob = pulp.LpProblem("Minimize_Energy", pulp.LpMinimize)

        # Decision variables: for each device, 1 if ON, 0 if OFF
        device_vars = {}
        for dev in self.devices:
            device_vars[dev['device_id']] = pulp.LpVariable(dev['device_id'], lowBound=0, upBound=1, cat='Binary')

        # Objective: minimize total energy consumption over the time interval
        prob += pulp.lpSum([dev['power_rating'] * self.dt * device_vars[dev['device_id']] for dev in self.devices])

        # Thermal dynamics: identify heater and cooler
        heater_id = None
        cooler_id = None
        for dev in self.devices:
            if dev['device_type'] == 'heater':
                heater_id = dev['device_id']
            elif dev['device_type'] == 'cooler':
                cooler_id = dev['device_id']

        # Temperature evolution constraint
        if heater_id and cooler_id:
            temp_next = (self.current_temp +
                         2.0 * device_vars[heater_id] -
                         2.0 * device_vars[cooler_id] -
                         0.1 * (self.current_temp - self.outside_temp))
        else:
            # If no heater/cooler, temperature unchanged
            temp_next = self.current_temp

        # Comfort constraints: temperature between 22 and 26
        prob += temp_next >= 22
        prob += temp_next <= 26

        # Occupancy comfort: if occupancy=1, lights must be on (assuming light devices)
        if self.occupancy == 1:
            for dev in self.devices:
                if dev['device_type'] == 'light':
                    prob += device_vars[dev['device_id']] == 1

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Extract results
        optimal_settings = {}
        for dev in self.devices:
            optimal_settings[dev['device_id']] = 'ON' if device_vars[dev['device_id']].varValue > 0.5 else 'OFF'

        return optimal_settings, pulp.value(prob.objective)


# =============================================================================
# PART 5: NLP RULE COMPILER (with trained intent model)
# =============================================================================
class NLPCompiler:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.intent_classifier = None
        self.intent_labels = ['set_device', 'unknown']
        self._train_intent_model()

    def _train_intent_model(self):
        # Create a small synthetic training set for intent detection
        train_texts = [
            "turn on the light when it gets dark",
            "switch on ac if temperature above 25",
            "set heater to on at 6pm",
            "turn off fan if no one home",
            "please activate the humidifier",
            "what is the weather",  # unknown
            "hello world",          # unknown
        ]
        train_intents = ['set_device', 'set_device', 'set_device', 'set_device', 'set_device', 'unknown', 'unknown']
        X = self.vectorizer.fit_transform(train_texts)
        self.intent_classifier = LogisticRegression()
        self.intent_classifier.fit(X, train_intents)

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        filtered = [w for w in tokens if w.isalnum() and w not in self.stop_words]
        return ' '.join(filtered)

    def detect_intent(self, text):
        clean = self.preprocess(text)
        X = self.vectorizer.transform([clean])
        intent = self.intent_classifier.predict(X)[0]
        return intent

    def extract_entities(self, text):
        # Rule-based extraction (could be replaced with a trained NER model)
        tokens = word_tokenize(text.lower())
        device_keywords = {'light', 'ac', 'heater', 'fan', 'tv', 'humidifier'}
        condition_keywords = {'temperature', 'humidity', 'occupancy', 'time'}
        device = None
        condition = None
        value = None
        time = None
        for i, tok in enumerate(tokens):
            if tok in device_keywords:
                device = tok
            if tok in condition_keywords:
                condition = tok
            if tok.isdigit():
                value = int(tok)
            if tok in ['at', 'after', 'before'] and i+1 < len(tokens) and tokens[i+1].isdigit():
                time = int(tokens[i+1])
        return device, condition, value, time

    def generate_rule(self, command):
        intent = self.detect_intent(command)
        if intent != 'set_device':
            return None
        device, condition, value, time = self.extract_entities(command)
        if device is None:
            return None
        # Default condition if not specified
        if condition is None:
            condition = 'temperature'
        if value is None:
            value = 22  # default threshold
        # Determine operator based on context (simplistic)
        operator = '>' if 'above' in command or 'higher' in command else '<' if 'below' in command or 'lower' in command else '=='
        # Determine action (on/off)
        action = 'ON' if 'on' in command else 'OFF' if 'off' in command else 'ON'
        rule = {
            'device_id': device,
            'condition_field': condition,
            'operator': operator,
            'threshold': value,
            'action': action,
            'priority': 5  # default priority
        }
        return rule

    def validate_rule(self, rule, existing_devices_df):
        if rule is None:
            return False
        # Check that device exists
        if rule['device_id'] not in existing_devices_df['device_id'].values:
            return False
        # Check condition field is valid
        if rule['condition_field'] not in ['temperature', 'humidity', 'occupancy']:
            return False
        # Additional checks (e.g., threshold range) could be added
        return True

    def compile(self, nl_commands_df, devices_df):
        new_rules = []
        for _, row in nl_commands_df.iterrows():
            rule = self.generate_rule(row['command'])
            if self.validate_rule(rule, devices_df):
                new_rules.append(rule)
        return pd.DataFrame(new_rules)


# =============================================================================
# PART 6: REINFORCEMENT LEARNING CONFLICT RESOLVER
# =============================================================================
class RLConflictResolver:
    """
    Q-learning agent that learns to choose among conflicting rules.
    Environment: simple simulation of home with temperature dynamics.
    State: (device_id, temperature_bin) where bins: low (<22), comfortable (22-26), high (>26)
    Action: index of the rule to apply (from the set of conflicting rules for that device)
    Reward: +1 if chosen rule leads to comfortable temperature after application, -1 otherwise.
    """
    def __init__(self, rules_df, devices_df, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.rules = rules_df.to_dict('records')
        self.devices = devices_df.to_dict('records')
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        # Build mapping of conflicting rule sets per device
        self.device_rules = defaultdict(list)
        for i, r in enumerate(self.rules):
            self.device_rules[r['device_id']].append(i)

    def _get_temp_bin(self, temp):
        if temp < 22:
            return 'low'
        elif temp <= 26:
            return 'comfortable'
        else:
            return 'high'

    def _get_state(self, device, temp):
        return (device, self._get_temp_bin(temp))

    def _get_possible_actions(self, device):
        return self.device_rules[device]

    def _simulate_environment(self, device, rule_idx, current_temp, occupancy):
        """Apply rule and compute next temperature and reward."""
        rule = self.rules[rule_idx]
        # Determine which device is affected (may be different from the rule's device)
        # In conflict, we are selecting a rule for a specific device, so rule.device_id == device
        # Simulate effect: if rule.action == 'ON' and device is heater, temp increases; if cooler, temp decreases.
        dev = next((d for d in self.devices if d['device_id'] == device), None)
        if dev is None:
            return current_temp, -1  # invalid device

        # Simple thermal effect based on device type
        new_temp = current_temp
        if dev['device_type'] == 'heater' and rule['action'] == 'ON':
            new_temp += 2.0
        elif dev['device_type'] == 'cooler' and rule['action'] == 'ON':
            new_temp -= 2.0
        # Natural drift towards 20
        new_temp -= 0.1 * (current_temp - 20)

        # Reward based on comfort after application
        if 22 <= new_temp <= 26:
            reward = 1
        else:
            reward = -1
        return new_temp, reward

    def train(self, episodes=500):
        for ep in range(episodes):
            # Random initial conditions
            device = random.choice(list(self.device_rules.keys()))
            current_temp = random.uniform(18, 30)
            occupancy = random.choice([0, 1])
            state = self._get_state(device, current_temp)
            possible_actions = self._get_possible_actions(device)
            if not possible_actions:
                continue

            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                action = random.choice(possible_actions)
            else:
                q_vals = {a: self.q_table[state][a] for a in possible_actions}
                action = max(q_vals, key=q_vals.get)

            # Simulate step
            next_temp, reward = self._simulate_environment(device, action, current_temp, occupancy)
            next_state = self._get_state(device, next_temp)
            next_possible_actions = self._get_possible_actions(device)

            # Q-learning update
            best_next = max([self.q_table[next_state][a] for a in next_possible_actions], default=0)
            td_target = reward + self.gamma * best_next
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * td_error

    def resolve(self, device, current_temp, occupancy):
        state = self._get_state(device, current_temp)
        possible_actions = self._get_possible_actions(device)
        if not possible_actions:
            return None
        q_vals = {a: self.q_table[state][a] for a in possible_actions}
        best_action = max(q_vals, key=q_vals.get)
        return self.rules[best_action]


# =============================================================================
# MASTER ALGORITHM
# =============================================================================
class SmartHomeDigitalTwinSystem:
    def __init__(self):
        # Generate synthetic datasets (in practice, load from CSV)
        self.devices_df = self._generate_devices()
        self.environment_df = self._generate_environment()
        self.rules_df = self._generate_rules()
        self.energy_df = self._generate_energy()
        self.nl_commands_df = self._generate_nl_commands()

    def _generate_devices(self):
        return pd.DataFrame([
            {'device_id': 'light1', 'state': 'OFF', 'power_rating': 10, 'device_type': 'light'},
            {'device_id': 'heater1', 'state': 'OFF', 'power_rating': 2000, 'device_type': 'heater'},
            {'device_id': 'cooler1', 'state': 'OFF', 'power_rating': 1500, 'device_type': 'cooler'},
        ])

    def _generate_environment(self):
        # 24 hours of simulated data
        times = range(24)
        temps = [20 + 5 * np.sin(i/24 * 2*np.pi) + random.uniform(-1,1) for i in times]
        hum = [50 + 10 * random.random() for _ in times]
        occ = [1 if 8 <= i <= 22 else 0 for i in times]
        return pd.DataFrame({'time': times, 'temperature': temps, 'humidity': hum, 'occupancy': occ})

    def _generate_rules(self):
        return pd.DataFrame([
            {'rule_id': 1, 'device_id': 'cooler1', 'condition_field': 'temperature', 'operator': '>', 'threshold': 26, 'action': 'ON', 'priority': 1},
            {'rule_id': 2, 'device_id': 'heater1', 'condition_field': 'temperature', 'operator': '<', 'threshold': 18, 'action': 'ON', 'priority': 2},
            {'rule_id': 3, 'device_id': 'light1', 'condition_field': 'occupancy', 'operator': '==', 'threshold': 1, 'action': 'ON', 'priority': 3},
            # Conflicting rule for cooler1 (same condition field, opposite action)
            {'rule_id': 4, 'device_id': 'cooler1', 'condition_field': 'temperature', 'operator': '>', 'threshold': 26, 'action': 'OFF', 'priority': 4},
        ])

    def _generate_energy(self):
        np.random.seed(42)
        values = np.cumsum(np.random.randn(100) * 0.5) + 10
        return pd.DataFrame({'timestamp': range(100), 'energy': values})

    def _generate_nl_commands(self):
        return pd.DataFrame([
            {'command': 'turn on the ac if temperature above 25'},
            {'command': 'switch off lights when no one home'},
            {'command': 'set heater to on at 6pm'}
        ])

    def run(self):
        print("="*60)
        print("SMART HOME DIGITAL TWIN SYSTEM")
        print("="*60)

        # 1. Digital Twin Simulation
        print("\n[1] Running Digital Twin Simulator...")
        sim = DigitalTwinSimulator(self.devices_df, self.environment_df, self.rules_df)
        sim_results = sim.run_simulation()
        print(f"    Total Energy: {sim.total_energy:.2f} Wh")
        print(f"    Comfort Score: {sim.comfort_score}")

        # 2. Conflict Detection
        print("\n[2] Running Conflict Detection...")
        detector = ConflictDetector(self.rules_df)
        resolved = detector.run()
        print(f"    Conflicts detected: {len(detector.detect_conflicts())}")
        print(f"    Resolved rules count: {len(resolved)}")
        for r in resolved:
            print(f"        Resolved rule: device={r['device_id']}, action={r['action']} (priority {r['priority']})")

        # 3. LSTM Energy Prediction
        print("\n[3] Training LSTM Energy Prediction Model...")
        predictor = LSTMEnergyPredictor(self.energy_df, sequence_length=10)
        rmse = predictor.train(epochs=20)
        last_seq = self.energy_df['energy'].values[-10:]
        next_pred = predictor.predict_next(last_seq)
        print(f"    Next energy prediction: {next_pred:.2f} Wh")

        # 4. Linear Programming Optimization
        print("\n[4] Running Linear Programming Optimization...")
        # Use current environment from last time step
        current_env = self.environment_df.iloc[-1]
        optimizer = LPOptimizer(self.devices_df, current_env['temperature'], current_env['occupancy'])
        opt_settings, min_energy = optimizer.optimize()
        print(f"    Optimal energy consumption: {min_energy:.2f} Wh")
        print("    Optimized device settings:")
        for dev, state in opt_settings.items():
            print(f"        {dev}: {state}")

        # 5. NLP Rule Compiler
        print("\n[5] Compiling Natural Language Commands...")
        nlp = NLPCompiler()
        new_rules_df = nlp.compile(self.nl_commands_df, self.devices_df)
        print(f"    Generated {len(new_rules_df)} new rule(s):")
        if not new_rules_df.empty:
            print(new_rules_df)
        else:
            print("    No valid rules generated.")

        # 6. RL Conflict Resolver
        print("\n[6] Training RL Conflict Resolver...")
        rl = RLConflictResolver(self.rules_df, self.devices_df)
        rl.train(episodes=500)
        # Resolve for a specific device and current conditions
        device = 'cooler1'
        current_temp = current_env['temperature']
        occupancy = current_env['occupancy']
        best_rule = rl.resolve(device, current_temp, occupancy)
        print(f"    RL selected rule for {device} at temp={current_temp:.1f}°C, occupancy={occupancy}:")
        print(f"        {best_rule}")

        print("\n" + "="*60)
        print("SYSTEM EXECUTION COMPLETE")
        print("="*60)


if __name__ == "__main__":
    system = SmartHomeDigitalTwinSystem()
    system.run()