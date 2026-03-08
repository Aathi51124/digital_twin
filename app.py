import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import pulp
import plotly.express as px
import plotly.graph_objects as go

# Download NLTK data (only once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# =============================================================================
# 1. DIGITAL TWIN SIMULATOR
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

        value = env.get(condition_field)
        if value is None:
            return

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
# 2. CONFLICT DETECTION (GRAPH)
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
                for idx in indices:
                    for jdx in indices:
                        if idx >= jdx:
                            continue
                        r1 = self.rules[idx]
                        r2 = self.rules[jdx]
                        if (r1['condition_field'] == r2['condition_field'] and
                            r1['action'] != r2['action']):
                            conflicts.append((idx, jdx))
        return conflicts

    def resolve_conflict(self, conflict_pair):
        idx, jdx = conflict_pair
        if self.rules[idx]['priority'] < self.rules[jdx]['priority']:
            return self.rules[idx]
        else:
            return self.rules[jdx]

    def run(self):
        self.build_graph()
        conflicts = self.detect_conflicts()
        resolved = [self.resolve_conflict(c) for c in conflicts]
        return resolved, conflicts


# =============================================================================
# 3. LSTM ENERGY PREDICTOR
# =============================================================================
class LSTMEnergyPredictor:
    def __init__(self, energy_df, sequence_length=10):
        self.energy = energy_df['energy'].values.astype(float)
        self.seq_len = sequence_length
        self.scaler_mean = None
        self.scaler_std = None
        self.model = None

    def preprocess(self):
        self.scaler_mean = self.energy.mean()
        self.scaler_std = self.energy.std()
        norm_vals = (self.energy - self.scaler_mean) / self.scaler_std
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
        split = int(len(X) * (1 - validation_split))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model = self.build_model((self.seq_len, 1))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                       epochs=epochs, batch_size=32, callbacks=[early_stop], verbose=0)
        y_pred = self.model.predict(X_test, verbose=0)
        rmse = np.sqrt(np.mean((y_pred.flatten() - y_test)**2))
        rmse_denorm = rmse * self.scaler_std
        return rmse_denorm

    def predict_next(self, last_sequence):
        last_norm = (np.array(last_sequence) - self.scaler_mean) / self.scaler_std
        last_norm = last_norm.reshape(1, self.seq_len, 1)
        pred_norm = self.model.predict(last_norm, verbose=0)[0,0]
        return pred_norm * self.scaler_std + self.scaler_mean


# =============================================================================
# 4. LINEAR PROGRAMMING OPTIMIZER
# =============================================================================
class LPOptimizer:
    def __init__(self, devices_df, current_temp, occupancy, time_interval=1.0):
        self.devices = devices_df.to_dict('records')
        self.current_temp = current_temp
        self.occupancy = occupancy
        self.dt = time_interval
        self.outside_temp = 20.0

    def optimize(self):
        prob = pulp.LpProblem("Minimize_Energy", pulp.LpMinimize)
        device_vars = {}
        for dev in self.devices:
            device_vars[dev['device_id']] = pulp.LpVariable(dev['device_id'], lowBound=0, upBound=1, cat='Binary')

        prob += pulp.lpSum([dev['power_rating'] * self.dt * device_vars[dev['device_id']] for dev in self.devices])

        # Identify heater and cooler
        heater_id = next((d['device_id'] for d in self.devices if d['device_type'] == 'heater'), None)
        cooler_id = next((d['device_id'] for d in self.devices if d['device_type'] == 'cooler'), None)

        if heater_id and cooler_id:
            temp_next = (self.current_temp +
                         2.0 * device_vars[heater_id] -
                         2.0 * device_vars[cooler_id] -
                         0.1 * (self.current_temp - self.outside_temp))
        else:
            temp_next = self.current_temp

        prob += temp_next >= 22
        prob += temp_next <= 26

        if self.occupancy == 1:
            for dev in self.devices:
                if dev['device_type'] == 'light':
                    prob += device_vars[dev['device_id']] == 1

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        optimal = {}
        for dev in self.devices:
            optimal[dev['device_id']] = 'ON' if device_vars[dev['device_id']].varValue > 0.5 else 'OFF'
        return optimal, pulp.value(prob.objective)


# =============================================================================
# 5. NLP RULE COMPILER (with trained intent)
# =============================================================================
class NLPCompiler:
    def __init__(self, devices_df):
        self.devices_df = devices_df
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.intent_classifier = None
        self._train_intent_model()
        # Mapping from common names to device IDs
        self.device_map = {
            'light': 'light1',
            'ac': 'cooler1',
            'heater': 'heater1',
            'cooler': 'cooler1',
            'fan': None,  # not in our devices
            'tv': None,
            'humidifier': None
        }

    def _train_intent_model(self):
        train_texts = [
            "turn on the light when it gets dark",
            "switch on ac if temperature above 25",
            "set heater to on at 6pm",
            "turn off fan if no one home",
            "please activate the humidifier",
            "what is the weather",
            "hello world",
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
        return self.intent_classifier.predict(X)[0]

    def extract_entities(self, text):
        tokens = word_tokenize(text.lower())
        device_keywords = set(self.device_map.keys())
        condition_keywords = {'temperature', 'humidity', 'occupancy', 'time'}
        device = None
        condition = None
        value = None
        time_val = None
        for i, tok in enumerate(tokens):
            if tok in device_keywords:
                device = self.device_map[tok]  # map to actual ID
            if tok in condition_keywords:
                condition = tok
            if tok.isdigit():
                value = int(tok)
            if tok in ['at', 'after', 'before'] and i+1 < len(tokens) and tokens[i+1].isdigit():
                time_val = int(tokens[i+1])
        return device, condition, value, time_val

    def generate_rule(self, command):
        intent = self.detect_intent(command)
        if intent != 'set_device':
            return None
        device, condition, value, time_val = self.extract_entities(command)
        if device is None:
            return None
        if condition is None:
            condition = 'temperature'
        if value is None:
            value = 22
        operator = '>' if 'above' in command or 'higher' in command else '<' if 'below' in command or 'lower' in command else '=='
        action = 'ON' if 'on' in command else 'OFF' if 'off' in command else 'ON'
        rule = {
            'device_id': device,
            'condition_field': condition,
            'operator': operator,
            'threshold': value,
            'action': action,
            'priority': 5
        }
        return rule

    def validate_rule(self, rule):
        if rule is None:
            return False
        if rule['device_id'] not in self.devices_df['device_id'].values:
            return False
        if rule['condition_field'] not in ['temperature', 'humidity', 'occupancy']:
            return False
        return True

    def compile(self, nl_commands_df):
        new_rules = []
        for _, row in nl_commands_df.iterrows():
            rule = self.generate_rule(row['command'])
            if self.validate_rule(rule):
                new_rules.append(rule)
        return pd.DataFrame(new_rules)


# =============================================================================
# 6. RL CONFLICT RESOLVER (Q-learning)
# =============================================================================
class RLConflictResolver:
    def __init__(self, rules_df, devices_df, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.rules = rules_df.to_dict('records')
        self.devices = devices_df.to_dict('records')
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
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
        rule = self.rules[rule_idx]
        dev = next((d for d in self.devices if d['device_id'] == device), None)
        if dev is None:
            return current_temp, -1
        new_temp = current_temp
        if dev['device_type'] == 'heater' and rule['action'] == 'ON':
            new_temp += 2.0
        elif dev['device_type'] == 'cooler' and rule['action'] == 'ON':
            new_temp -= 2.0
        new_temp -= 0.1 * (current_temp - 20)
        reward = 1 if 22 <= new_temp <= 26 else -1
        return new_temp, reward

    def train(self, episodes=500):
        for _ in range(episodes):
            device = random.choice(list(self.device_rules.keys()))
            current_temp = random.uniform(18, 30)
            occupancy = random.choice([0, 1])
            state = self._get_state(device, current_temp)
            possible = self._get_possible_actions(device)
            if not possible:
                continue
            if random.random() < self.epsilon:
                action = random.choice(possible)
            else:
                q_vals = {a: self.q_table[state][a] for a in possible}
                action = max(q_vals, key=q_vals.get)
            next_temp, reward = self._simulate_environment(device, action, current_temp, occupancy)
            next_state = self._get_state(device, next_temp)
            next_possible = self._get_possible_actions(device)
            best_next = max([self.q_table[next_state][a] for a in next_possible], default=0)
            td_target = reward + self.gamma * best_next
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * td_error

    def resolve(self, device, current_temp, occupancy):
        state = self._get_state(device, current_temp)
        possible = self._get_possible_actions(device)
        if not possible:
            return None
        q_vals = {a: self.q_table[state][a] for a in possible}
        best_action = max(q_vals, key=q_vals.get)
        return self.rules[best_action]


# =============================================================================
# DATA GENERATION (cached)
# =============================================================================
@st.cache_data
def generate_data():
    devices = pd.DataFrame([
        {'device_id': 'light1', 'state': 'OFF', 'power_rating': 10, 'device_type': 'light'},
        {'device_id': 'heater1', 'state': 'OFF', 'power_rating': 2000, 'device_type': 'heater'},
        {'device_id': 'cooler1', 'state': 'OFF', 'power_rating': 1500, 'device_type': 'cooler'},
    ])
    # 24 time steps
    times = list(range(24))
    temps = [20 + 5 * np.sin(i/24 * 2*np.pi) + random.uniform(-1,1) for i in times]
    hum = [50 + 10 * random.random() for _ in times]
    occ = [1 if 8 <= i <= 22 else 0 for i in times]
    env = pd.DataFrame({'time': times, 'temperature': temps, 'humidity': hum, 'occupancy': occ})
    rules = pd.DataFrame([
        {'rule_id': 1, 'device_id': 'cooler1', 'condition_field': 'temperature', 'operator': '>', 'threshold': 26, 'action': 'ON', 'priority': 1},
        {'rule_id': 2, 'device_id': 'heater1', 'condition_field': 'temperature', 'operator': '<', 'threshold': 18, 'action': 'ON', 'priority': 2},
        {'rule_id': 3, 'device_id': 'light1', 'condition_field': 'occupancy', 'operator': '==', 'threshold': 1, 'action': 'ON', 'priority': 3},
        {'rule_id': 4, 'device_id': 'cooler1', 'condition_field': 'temperature', 'operator': '>', 'threshold': 26, 'action': 'OFF', 'priority': 4},
    ])
    np.random.seed(42)
    energy_vals = np.cumsum(np.random.randn(100) * 0.5) + 10
    energy = pd.DataFrame({'timestamp': range(100), 'energy': energy_vals})
    nl_commands = pd.DataFrame([
        {'command': 'turn on the ac if temperature above 25'},
        {'command': 'switch off lights when no one home'},
        {'command': 'set heater to on at 6pm'}
    ])
    return devices, env, rules, energy, nl_commands

# =============================================================================
# STREAMLIT APP
# =============================================================================
st.set_page_config(layout="wide", page_title="Smart Home Digital Twin")
st.title("🏠 Smart Home Digital Twin Dashboard")

# Sidebar controls
st.sidebar.header("Simulation Controls")
time_interval = st.sidebar.slider("Time interval (hours)", 0.5, 2.0, 1.0, 0.1)
run_button = st.sidebar.button("Run Full Simulation")

# Load data
devices_df, env_df, rules_df, energy_df, nl_df = generate_data()

# Initialize modules (cached heavy training)
@st.cache_resource
def train_lstm(energy_df):
    pred = LSTMEnergyPredictor(energy_df, sequence_length=10)
    rmse = pred.train(epochs=20)
    return pred, rmse

@st.cache_resource
def train_rl(rules_df, devices_df):
    rl = RLConflictResolver(rules_df, devices_df)
    rl.train(episodes=500)
    return rl

# Train once and store in session state to avoid re-running on every interaction
if 'lstm_predictor' not in st.session_state:
    st.session_state.lstm_predictor, st.session_state.lstm_rmse = train_lstm(energy_df)
if 'rl_resolver' not in st.session_state:
    st.session_state.rl_resolver = train_rl(rules_df, devices_df)

# Main area with tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Simulation", "Conflict Detection", "Energy Prediction",
    "Optimization", "NLP Compiler", "RL Resolver", "Summary"
])

with tab1:
    st.header("Digital Twin Simulation")
    if run_button:
        with st.spinner("Running simulation..."):
            sim = DigitalTwinSimulator(devices_df, env_df, rules_df)
            sim_results = sim.run_simulation(time_interval=time_interval)
        st.success("Simulation complete!")
        st.metric("Total Energy (Wh)", f"{sim.total_energy:.2f}")
        st.metric("Comfort Score", f"{sim.comfort_score}")
        
        # Plot temperature and energy over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim_results['time_step'], y=sim_results['temperature'],
                                  mode='lines+markers', name='Temperature'))
        fig.add_trace(go.Scatter(x=sim_results['time_step'], y=sim_results['energy_step'],
                                  mode='lines+markers', name='Energy Step', yaxis='y2'))
        fig.update_layout(
            title='Environmental and Energy Data',
            xaxis_title='Time Step',
            yaxis_title='Temperature (°C)',
            yaxis2=dict(title='Energy (Wh)', overlaying='y', side='right')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Device states heatmap
        device_names = [d['id'] for d in sim.virtual_home]
        states_matrix = []
        for res in sim_results.to_dict('records'):
            states = [1 if s == 'ON' else 0 for s in res['device_states']]
            states_matrix.append(states)
        fig2 = px.imshow(np.array(states_matrix).T,
                         x=sim_results['time_step'],
                         y=device_names,
                         color_continuous_scale='Viridis',
                         title='Device States (ON=1, OFF=0)')
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(sim_results)
    else:
        st.info("Click 'Run Full Simulation' in the sidebar to start.")

with tab2:
    st.header("Conflict Detection (Rule Dependency Graph)")
    detector = ConflictDetector(rules_df)
    resolved_rules, conflicts = detector.run()
    st.subheader("Detected Conflicts")
    if conflicts:
        for (i, j) in conflicts:
            r1 = rules_df.iloc[i]
            r2 = rules_df.iloc[j]
            st.warning(f"Conflict between Rule {r1['rule_id']} and Rule {r2['rule_id']} on device {r1['device_id']}")
    else:
        st.success("No conflicts detected.")

    st.subheader("Resolved Rules (by priority)")
    if resolved_rules:
        st.dataframe(pd.DataFrame(resolved_rules))
    else:
        st.info("No rules needed resolution.")

    # Show rule graph (simple table)
    st.subheader("Rule Graph (devices and rule indices)")
    graph_df = pd.DataFrame([
        {'device': dev, 'rule_indices': idxs} for dev, idxs in detector.graph.items()
    ])
    st.dataframe(graph_df)

with tab3:
    st.header("LSTM Energy Prediction")
    st.metric("Training RMSE (denormalized)", f"{st.session_state.lstm_rmse:.2f} Wh")
    last_10 = energy_df['energy'].values[-10:]
    next_pred = st.session_state.lstm_predictor.predict_next(last_10)
    st.metric("Next Energy Prediction", f"{next_pred:.2f} Wh")

    # Plot historical and prediction
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energy_df['timestamp'], y=energy_df['energy'],
                              mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=[energy_df['timestamp'].iloc[-1] + 1], y=[next_pred],
                              mode='markers', marker=dict(size=10, color='red'), name='Next Prediction'))
    fig.update_layout(title='Energy Consumption History and Next Prediction',
                      xaxis_title='Time', yaxis_title='Energy (Wh)')
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Linear Programming Optimization")
    # Use last environment state
    last_env = env_df.iloc[-1]
    st.write(f"Current temperature: {last_env['temperature']:.1f}°C, Occupancy: {'Yes' if last_env['occupancy'] else 'No'}")
    opt = LPOptimizer(devices_df, last_env['temperature'], last_env['occupancy'], time_interval)
    opt_settings, min_energy = opt.optimize()
    st.metric("Minimum Possible Energy (Wh)", f"{min_energy:.2f}")
    st.subheader("Optimal Device Settings")
    for dev, state in opt_settings.items():
        st.write(f"**{dev}**: {state}")

with tab5:
    st.header("NLP Rule Compiler")
    st.subheader("Natural Language Commands")
    st.dataframe(nl_df)
    nlp = NLPCompiler(devices_df)
    new_rules = nlp.compile(nl_df)
    st.subheader("Generated Rules")
    if not new_rules.empty:
        st.dataframe(new_rules)
    else:
        st.info("No valid rules generated. Check device mapping or command phrasing.")

    # Show intent detection example
    st.subheader("Intent Detection Example")
    example = st.text_input("Enter a command to test:", "turn on the heater when temperature below 20")
    if example:
        intent = nlp.detect_intent(example)
        st.write(f"Detected intent: **{intent}**")
        if intent == 'set_device':
            rule = nlp.generate_rule(example)
            st.write("Generated rule:", rule)

with tab6:
    st.header("RL Conflict Resolver")
    st.write("Q-learning agent trained on simulated environment.")
    # Show Q-table summary (first few entries)
    st.subheader("Q-table sample")
    q_sample = []
    for (state, actions) in list(st.session_state.rl_resolver.q_table.items())[:5]:
        for act, val in actions.items():
            q_sample.append({'state': state, 'action': act, 'Q-value': val})
    if q_sample:
        st.dataframe(pd.DataFrame(q_sample))
    else:
        st.info("Q-table is empty (maybe no training done yet).")

    # Resolve for current conditions
    st.subheader("Resolve Conflict for a Device")
    device_choice = st.selectbox("Select device", devices_df['device_id'].tolist())
    temp_input = st.slider("Current temperature", 15.0, 35.0, 22.0)
    occ_input = st.checkbox("Occupied")
    if st.button("Resolve"):
        best_rule = st.session_state.rl_resolver.resolve(device_choice, temp_input, occ_input)
        if best_rule:
            st.success("Optimal rule according to Q-learning:")
            st.json(best_rule)
        else:
            st.warning("No conflicting rules for this device or state.")

with tab7:
    st.header("System Summary")
    st.subheader("Datasets")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Devices**")
        st.dataframe(devices_df)
    with col2:
        st.write("**Environment (first 5)**")
        st.dataframe(env_df.head())
    with col3:
        st.write("**Rules**")
        st.dataframe(rules_df)

    st.subheader("Overall Metrics")
    if run_button and 'sim' in locals():
        st.metric("Total Energy", f"{sim.total_energy:.2f} Wh")
        st.metric("Comfort Score", f"{sim.comfort_score}")
    else:
        st.info("Run simulation to see metrics.")