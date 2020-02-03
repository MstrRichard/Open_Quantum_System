import numpy as np
class Gate:

    def __init__(self, gate_type, sites, exp=[0.0,0.0]):

        self.type = gate_type
        self.sites = sites
        self.exp = exp
        self.nb_qbits_affected = len(sites)

    def __repr__(self):
        return self.type + " " + " ".join([str(s) for s in self.sites]) + " " + " ".join([str(s) for s in self.exp])


class QuantumCircuit:

    def __init__(self, nb_qbits=-1, gates=[]):

        self.nb_qbits = nb_qbits
        self.gates = gates

    @classmethod
    def from_file(cls, file_path):

        gates = []

        with open(file_path, 'r') as file:

            nb_qbits = int(file.readline())

            for line in file:
                line = line.split()
                gates.append(Gate(line[0], [int(s) for s in line[1:]]))

        return cls(nb_qbits, gates)

    def to_file(self, file_path):

        with open(file_path, 'w') as file:

            file.write(str(self.nb_qbits) + "\n")

            for gate in self.gates:
                file.write(str(gate) + "\n")

    def __getitem__(self, item):
        return self.gates[item]

    def __len__(self):
        return len(self.gates)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            gate = self.gates[self.n]
            self.n += 1
            return gate
        else:
            raise StopIteration


class BasicCircuit(QuantumCircuit):

    def __init__(self, nb_qbits):

        gates = [Gate("CZ", [i, i+1]) for i in range(nb_qbits - 1)]

        super(BasicCircuit, self).__init__(nb_qbits, gates)


class GHZCircuit(QuantumCircuit):

    def __init__(self, nb_qbits):

        gates = [Gate("HI", [0, 1])]
        gates += [Gate("CNot", [i, i+1]) for i in range(nb_qbits - 1)]

        super(GHZCircuit, self).__init__(nb_qbits, gates)


class VQE_GHZ(QuantumCircuit):

    def __init__(self, nb_qbits):

        seq = np.zeros(nb_qbits)
        if nb_qbits == 8:
            seq = np.array([0.5297, 0.5243, 0.7243, 0.6151, 0.6151, 0.7243, 0.5243, 0.5297]) * (-1j)
        elif nb_qbits == 10:
            seq = np.array([0.5814, 0.5230, 0.6360, 0.7889, 0.5993, 0.5993, 0.7889, 0.6360, 0.5230, 0.5814]) * (-1j)
        elif nb_qbits == 12 or nb_qbits == 4:
            seq = np.array([0.5466, 0.5452, 0.6902,0.7212,0.5946,0.7276,0.7276,0.5946,0.7212,0.6902,0.5452,0.5466]) * (-1j)

        gamma_seq = seq[::2]
        beta_seq = seq[1::2]

        gates = []
        for i in range(len(gamma_seq)):
            for j in range(nb_qbits//2):
                gates += [Gate("exp_ZZ", [2*j, 2*j+1], [gamma_seq[i], 0.])]
            for j in range(nb_qbits//2-1):
                gates += [Gate("exp_ZZX", [2*j+1, 2*j+2], [gamma_seq[i], beta_seq[i]])]
            gates += [Gate("exp_ZZX", [nb_qbits-1, 0], [gamma_seq[i], beta_seq[i]])]

        super(VQE_GHZ, self).__init__(nb_qbits, gates)


class VQE_TFIM(QuantumCircuit):

    def __init__(self, nb_qbits):

        seq = np.zeros(nb_qbits)

        if nb_qbits == 8 or nb_qbits == 6:
            seq = np.array([0.2496,0.6845,0.4808,0.6559,0.5260,0.6048,0.4503,0.3180]) * (-1j)
        elif nb_qbits == 10:
            seq = np.array([0.2473, 0.6977,0.4888,0.6783,0.5559,0.6567,0.5558,0.6029,0.4598,0.3068]) * (-1j)
        elif nb_qbits == 12 or nb_qbits == 4:
            seq = np.array([0.2809,0.6131,0.6633,0.4537,0.8653,0.4663,0.6970,0.6829,0.4569,0.7990,0.3565,0.4304]) * (-1j)

        gamma_seq = seq[::2]
        beta_seq = seq[1::2]

        gates = []
        for i in range(len(gamma_seq)):
            for j in range(nb_qbits//2):
                gates += [Gate("exp_ZZ", [2*j, 2*j+1], [gamma_seq[i], 0.])]
            for j in range(nb_qbits//2-1):
                gates += [Gate("exp_ZZX", [2*j+1, 2*j+2], [gamma_seq[i], beta_seq[i]])]
            gates += [Gate("exp_ZZX", [nb_qbits-1, 0], [gamma_seq[i], beta_seq[i]])]

        super(VQE_TFIM, self).__init__(nb_qbits, gates)


class VQE_TFIM_full(QuantumCircuit):

    def __init__(self, nb_qbits):

        seq = np.zeros(nb_qbits)

        if nb_qbits == 8 or nb_qbits == 6:
            seq = np.array([0.2496,0.6845,0.4808,0.6559,0.5260,0.6048,0.4503,0.3180]) * (-1j)
        elif nb_qbits == 10:
            seq = np.array([0.2473, 0.6977,0.4888,0.6783,0.5559,0.6567,0.5558,0.6029,0.4598,0.3068]) * (-1j)
        elif nb_qbits == 12 or nb_qbits == 4:
            seq = np.array([0.2809,0.6131,0.6633,0.4537,0.8653,0.4663,0.6970,0.6829,0.4569,0.7990,0.3565,0.4304]) * (-1j)

        gamma_seq = seq[::2]
        beta_seq = seq[1::2]

        gates = []
        for i in range(len(gamma_seq)):
            for j in range(nb_qbits):
                gates += [Gate("exp_ZZ", [j%nb_qbits, (j+1)%nb_qbits], [gamma_seq[i], 0.])]
            for j in range(nb_qbits):
                gates += [Gate("exp_XI", [j%nb_qbits, (j+1)%nb_qbits], [gamma_seq[i], beta_seq[i]])]

        super(VQE_TFIM_full, self).__init__(nb_qbits, gates)


class RandomCircuit(QuantumCircuit):

    def __init__(self, step, nb_qbits):

        #np.random.seed(8)
        gate_type = ["CNot", "PhaseI", "TI", "HI"]
        type_rand = np.random.randint(len(gate_type), size=step)
        pos_rand = np.random.randint(nb_qbits-1, size=step)
        gates = [Gate(gate_type[type_rand[i]], [pos_rand[i], pos_rand[i]+1]) for i in range(step)]
        #gates = [Gate("PhaseI", [0,1])]

        super(RandomCircuit, self).__init__(nb_qbits, gates)


class BasicCircuit_step(QuantumCircuit):

    def __init__(self, step, nb_qbits):

        gates = [Gate("CZ", [i, i+1]) for i in range(step+1, nb_qbits - 1)]

        super(BasicCircuit_step, self).__init__(nb_qbits, gates)


def get_circuit(circuit_config):
    """
    Get the corresponding circuit for the configurations, note that file precedes
    :param circuit_config:
    :return:
    """
    if circuit_config.circuit_file:
        return QuantumCircuit.from_file(circuit_config.circuit_file)

    if circuit_config.circuit_type == "basic":

        assert circuit_config.nb_qbits != -1, "The number of qbits: {} should be a positive integer"
        return BasicCircuit(circuit_config.nb_qbits)

    elif circuit_config.circuit_type == "GHZ":

        assert circuit_config.nb_qbits != -1, "The number of qbits: {} should be a positive integer"
        return GHZCircuit(circuit_config.nb_qbits)
    elif circuit_config.circuit_type == "VQE_GHZ":

        assert circuit_config.nb_qbits != -1, "The number of qbits: {} should be a positive integer"
        return VQE_GHZ(circuit_config.nb_qbits)
    elif circuit_config.circuit_type == "VQE_TFIM":

        assert circuit_config.nb_qbits != -1, "The number of qbits: {} should be a positive integer"
        return VQE_TFIM(circuit_config.nb_qbits)
    elif circuit_config.circuit_type == "VQE_TFIM_full":

        assert circuit_config.nb_qbits != -1, "The number of qbits: {} should be a positive integer"
        return VQE_TFIM_full(circuit_config.nb_qbits)
