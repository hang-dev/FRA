from circuit_knitting.cutting.cutqc.wire_cutting import _generate_metadata, _attribute_shots
from qiskit.circuit.random import random_circuit
import numpy as np
import copy
import networkx as nx
import math
from qiskit import QuantumCircuit, Aer, transpile
from circuit_knitting.cutting.cutqc import (
    cut_circuit_wires,
    evaluate_subcircuits,
)
from scipy.optimize import minimize

def maxcut_obj(solution, graph):
    """Given a bit string as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph."""
    obj = 0
    for i, j in graph.edges():
        if solution[i] != solution[j]:
            obj -= 1
    return obj


def compute_expectation(counts, graph):
    """Computes expectation value based on measurement results"""
    avg = 0
    sum_count = 0
    for bit_string, count in counts.items():
        obj = maxcut_obj(bit_string, graph)
        avg += obj * count
        sum_count += count
    return avg / sum_count


def create_qaoa_circ(graph, theta):
    """Creates a parametrized qaoa circuit"""
    nqubits = len(graph.nodes())
    n_layers = len(theta) // 2  # number of alternating unitaries
    beta = theta[:n_layers]
    gamma = theta[n_layers:]

    qc = QuantumCircuit(nqubits)

    # initial_state
    qc.h(range(nqubits))

    for layer_index in range(n_layers):
        # problem unitary
        for pair in list(graph.edges()):
            qc.rzz(2 * gamma[layer_index], pair[0], pair[1])
        # mixer unitary
        for qubit in range(nqubits):
            qc.rx(2 * beta[layer_index], qubit)

    qc.measure_all()
    return qc


def get_expectation(graph, shots=512):
    """Get expectation of QAOA circuit"""
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots

    def execute_circ(theta):
        qc = create_qaoa_circ(graph, theta)
        counts = backend.run(qc, seed_simulator=10,
                             nshots=512).result().get_counts()
        return compute_expectation(counts, graph)

    return execute_circ


def gen_random_circuit(n):
    """Generate random circuit"""
    nA = n // 2
    nB = n // 2
    A = random_circuit(nA, nA)
    B = random_circuit(nA, nB)
    C = random_circuit(nB - 1, nA)
    D = random_circuit(nB, nB)
    E = random_circuit(1, nA)

    circ = QuantumCircuit(nA + nB).compose(A)
    circ.compose(E, qubits=[nA], inplace=True)
    circ.compose(C, qubits=list(range(nA + 1, nA + nB)), inplace=True)
    circ.cnot(nA - 1, nA)
    circ.compose(B, qubits=list(range(0, nA)), inplace=True)
    circ.compose(D, qubits=list(range(nA, nA + nB)), inplace=True)
    circ.measure_all()

    subcirc1 = QuantumCircuit(nA + 1).compose(A)
    subcirc1.compose(E, qubits=[nA], inplace=True)
    subcirc1.cnot(nA - 1, nA)
    subcirc1.compose(B, qubits=list(range(0, nA)), inplace=True)

    subcirc2 = QuantumCircuit(nB).compose(D)



    return circ, subcirc1, subcirc2


def gen_QAOA_circuit(n):
    # Generate qaoa circuit
    G = nx.Graph()
    G.add_edges_from([(i, i + 1) for i in range(n - 1)], weight=1)
    expectation = get_expectation(G)
    res = minimize(expectation,
                   [1.0] * 2,
                   method='COBYLA')
    QAOA = create_qaoa_circ(G, res.x)

    beta = res.x[:1][0]
    gamma = res.x[1:][0]

    nA = math.ceil(n / 2)
    subcirc1 = QuantumCircuit(nA)
    subcirc1.h([i for i in range(nA)])
    for i in range(nA):
        if i <= nA - 2:
            subcirc1.rzz(2 * beta, i, i + 1)

    subcirc1.rx(2 * gamma, [i for i in range(nA - 1)])

    nB = n - nA + 1
    subcirc2 = QuantumCircuit(nB)
    subcirc2.h([i for i in range(1, nB)])
    for i in range(nB):
        if i <= nB - 2:
            subcirc2.rzz(2 * beta, i, i + 1)
    subcirc2.rx(2 * gamma, [i for i in range(nB)])
    # print('full circ:', circ)
    #
    # print('subcirc1:', subcirc1)
    #
    # print('subcirc2:', subcirc2)

    return QAOA, subcirc1, subcirc2


def MSE(target, obs, num_qubit):
    """  Compute the Mean Squared Error (MSE)"""
    target_list = [0 for i in range(2 ** num_qubit)]
    for key, value in target.items():
        target_list[int(key, 2)] = value
    target = np.array(target_list)
    obs_list = [0 for i in range(2 ** num_qubit)]
    for key, value in obs.items():
        obs_list[int(key, 2)] = value
    obs = np.array(obs_list)
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    if isinstance(target, dict):
        se = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            se += (t - o) ** 2
        mse = se / len(obs)
    elif isinstance(target, np.ndarray) and isinstance(obs, np.ndarray):
        target = target.reshape(-1, 1)
        obs = obs.reshape(-1, 1)
        squared_diff = (target - obs) ** 2
        se = np.sum(squared_diff)
        mse = np.mean(squared_diff)
    elif isinstance(target, np.ndarray) and isinstance(obs, dict):
        se = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = target[o_idx]
            se += (t - o) ** 2
        mse = se / len(obs)
    else:
        raise Exception("target type : %s" % type(target))
    return mse


def dict_to_array(distribution_dict, force_prob=True):
    """function of dict to array"""
    state = list(distribution_dict.keys())[0]
    num_qubits = len(state)
    num_shots = sum(distribution_dict.values())
    cnts = np.zeros(2**num_qubits, dtype=float)
    for state in distribution_dict:
        cnts[int(state, 2)] = distribution_dict[state]
    if abs(sum(cnts) - num_shots) > 1:
        print(
            "dict_to_array may be wrong, converted counts = {}, input counts = {}".format(
                sum(cnts), num_shots
            )
        )
    if not force_prob:
        return cnts
    else:
        prob = cnts / num_shots
        assert abs(sum(prob) - 1) < 1e-10
        return prob


def reconstruct_entry_distribution(subcircuit_instance_probabilities,cuts,num_threads: int = 1) :
    """compute subcircuit entry distribution"""
    summation_terms, subcircuit_entries, _ = _generate_metadata(cuts)

    subcircuit_entry_probabilities = _attribute_shots(
        subcircuit_entries, subcircuit_instance_probabilities
    )


    return subcircuit_entry_probabilities


def get_enrty_distribution(circ):
    """cut circuit and return subcircuit entry distribution"""
    circ = circ.remove_final_measurements(inplace=False)
    cuts = cut_circuit_wires(
        circuit=circ,
        method="automatic",
        max_subcircuit_width=math.floor(circ.num_qubits / 2) + 1,
        max_subcircuit_cuts=10,
        max_subcircuit_size=12,
        max_cuts=1,
        num_subcircuits=[2],
    )
    subcircuit_instance_probabilities = evaluate_subcircuits(cuts)
    subcircuit_entry_distributions =reconstruct_entry_distribution(subcircuit_instance_probabilities, cuts)

    return subcircuit_entry_distributions,cuts
















