import streamlit as st
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Introduction
st.title("Quantum Entanglement Interactive Guide 1")
st.write("""
In this interactive guide, you will explore quantum entanglement and its effects on measurement. Quantum entanglement is a phenomenon where particles become linked, such that the state of one particle directly affects the state of another, regardless of distance.

### Instructions:
1. **Select Experiment Type**: Choose between "Not Entangled" and "Entangled" states.
2. **Measurement**: Set measurement angles for the qubits.
3. **Observation**: See how the results of measurements are correlated or not.

Let's get started!
""")

# Options for Experiment Type
experiment_type = st.selectbox("Select Experiment Type", ["Not Entangled", "Entangled"])

# Sidebar for Measurement Angles
angle_1 = st.sidebar.slider("Set Measurement Angle for Qubit 1 (in degrees)", 0, 180, 90)
angle_2 = st.sidebar.slider("Set Measurement Angle for Qubit 2 (in degrees)", 0, 180, 90)

def create_not_entangled():
    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])
    return qc

def create_entangled():
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Apply Hadamard gate to the first qubit
    qc.cx(0, 1)  # Apply CNOT gate to entangle qubits
    return qc

def measure_experiment():
    if experiment_type == "Not Entangled":
        qc = create_not_entangled()
    elif experiment_type == "Entangled":
        qc = create_entangled()
        # Apply rotation gates to measure in different bases
        rad_angle_1 = angle_1 * np.pi / 180
        rad_angle_2 = angle_2 * np.pi / 180
        qc.ry(rad_angle_1, 0)
        qc.ry(rad_angle_2, 1)
        qc.measure([0, 1], [0, 1])
    
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=128, memory=True).result()
    memory = result.get_memory()
    counts = result.get_counts()
    return counts, qc, memory

def extract_qubit_results(memory):
    qubit_1_states = []
    qubit_2_states = []
    
    for shot in memory:
        qubit_1_states.append(int(shot[1]))  # Qubit 1 (second bit in '10' or '01')
        qubit_2_states.append(int(shot[0]))  # Qubit 2 (first bit in '10' or '01')
    
    return qubit_1_states, qubit_2_states

# Button to run the experiment
if st.button("Submit Measurements"):
    st.write("### Running Experiment")
    counts, qc, memory = measure_experiment()
    
    st.write(f"Selected Experiment Type: {experiment_type}")
    st.write(f"Qubit 1's measurement angle: {angle_1} degrees")
    st.write(f"Qubit 2's measurement angle: {angle_2} degrees")
    
    # Display Quantum Circuit
    st.write("### Quantum Circuit")
    st.write("The quantum circuit used in the simulation:")
    st.write(qc.draw(output='text'))

    # Separate counts for each qubit
    counts_0 = {'0': counts.get('00', 0) + counts.get('01', 0),
                '1': counts.get('10', 0) + counts.get('11', 0)}
    counts_1 = {'0': counts.get('00', 0) + counts.get('10', 0),
                '1': counts.get('01', 0) + counts.get('11', 0)}

    

    # Comparison of Results
    st.write("### Combined Measurement Results")
    st.write("The histogram below shows the combined measurement outcomes of both qubits:")
    fig, ax = plt.subplots()
    plot_histogram(counts, ax=ax, legend=['Measurement Outcomes'])
    st.pyplot(fig)


    qubit_1_states, qubit_2_states = extract_qubit_results(memory)
    results_matrix = np.array([qubit_1_states, qubit_2_states])


    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(results_matrix, cmap="Blues", cbar=False, linewidths=0.5, annot=True, fmt="d", ax=ax)
    
    ax.set_xlabel("Shot Number")
    ax.set_ylabel("Qubit (1 on top, 2 on bottom)")
    ax.set_title("Heatmap of Qubit Measurements Across Shots")

    st.pyplot(fig)

    # Extract and Display Separate Counts
    st.write("### Individual Qubit Measurement Counts")
    counts_00 = counts.get('00', 0)
    counts_01 = counts.get('01', 0)
    counts_10 = counts.get('10', 0)
    counts_11 = counts.get('11', 0)
    
    st.write(f"Counts for |00⟩: {counts_00}")
    st.write(f"Counts for |01⟩: {counts_01}")
    st.write(f"Counts for |10⟩: {counts_10}")
    st.write(f"Counts for |11⟩: {counts_11}")


footer="""<style>

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
text-align: center;
}
</style>

<div class="footer">
<p>"""rainbow:[Developed with :heart: by Raghav]"""</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
