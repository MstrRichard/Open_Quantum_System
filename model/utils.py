import torch
import itertools as it
from itertools import product
from utils import ncon
import numpy as np  # TODO: We probably can get rid on numpy
import scipy as sp
import tensorly as tl
from tensorly.decomposition import matrix_product_state



def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.relu, "swish": swish}


def vectorize(num_sites, K, model, device=torch.device("cpu")):
    """
    :return: a vector of probability distribution of a model over all the basis elements
    """
    #l_basis = []
    l_basis = np.zeros((K**num_sites, num_sites), dtype=np.int32)
    for i in range(K**num_sites):
      basis_str = np.base_repr(i, base=K, padding=num_sites)[-num_sites:]
      #l_basis.append(np.array(list(basis_str), dtype=int))
      l_basis[i, :] = np.array(list(basis_str), dtype=int)
    l_basis = torch.tensor(l_basis, dtype=torch.long, device=device)
    #l_basis = torch.tensor(l_basis)
    #l_basis.to(torch.device("cuda:0"))
    #lP = model.logP(l_basis, look_ahead=True, device=device)
    #lP = model.logP(l_basis, look_ahead=True)
    lP = model(forward_type='logP', seq=l_basis)
    return lP.cpu().numpy()


def prob_samples(prob, num_sites, num_samples, device=torch.device("cpu")):
    """
    :return: exact sampling of configurations from prob
    """
    configs = torch.multinomial(torch.tensor(prob), num_samples, replacement=True)
    samples = np.zeros((num_samples, num_sites), dtype=np.int32)
    for i in range(num_samples):
      basis_str = np.base_repr(configs[i].numpy(), base=4, padding=num_sites)[-num_sites:]
      samples[i, :] = np.array(list(basis_str), dtype=int)
    samples = torch.tensor(samples, dtype=torch.long, device=device)
    return configs.numpy(), samples


def prob_basis(num_sites, K=4):
    """
    :return: basis for povm probability distribution
    """
    l_basis = []
    for i in range(K**num_sites):
      basis_str = np.base_repr(i, base=K, padding=num_sites)[-num_sites:]
      l_basis.append(np.array(list(basis_str), dtype=int))
      #l_basis.append(basis_str)
    l_basis = torch.tensor(l_basis)
    return l_basis


def mps_to_vector(mps):
    """
    mps should have site more than 2
    :return: vector form of the mps
    """
    site = len(mps)
    psi = ncon((mps[0],mps[1]),([-1,1],[1,-2,-3]))
    psi = np.reshape(psi,[4,-1])
    for i in range(site-3):
        psi = ncon((psi,mps[i+2]),([-1,1],[1,-2,-3]))
        psi = np.reshape(psi,[2**(i+3),-1])
    psi = ncon((psi,mps[site-1]),([-1,1],[-2,1]))
    psi = np.reshape(psi,[2**site])
    return psi


def tl_to_mps(mps):
    """
    turn tensorly mps into Juan's mps form
    :return: Juan's mps form
    """
    mps[0] = np.reshape(mps[0],(2,2))
    mps[-1] = np.transpose(np.reshape(mps[-1],(2,-1)))
    return mps


def vector_to_mps(psi, N):
    """
    N: number of qubits
    :return: Juan's mps form of a vector
    """
    t1 = tl.tensor(psi.reshape([2]*N))
    rank = list(np.power(2,np.where( np.arange(1,N)<N/2,  np.arange(1,N), N-np.arange(1,N)))) # rank of the factors in MPS
    rank.insert(0, 1)
    rank.insert(N, 1)
    t1 = matrix_product_state(t1, rank)
    mps_t1 = tl_to_mps(t1)
    return mps_t1

def mps_to_density(mps):
    site = len(mps)
    #for i in range(site):
    #    print(mps[i])
    #assert False, 'stop'
    psi = ncon((mps[0],mps[1]),([-1,1],[1,-2,-3]))
    psi = np.reshape(psi,[4,-1])
    for i in range(site-3):
        psi = ncon((psi,mps[i+2]),([-1,1],[1,-2,-3]))
        psi = np.reshape(psi,[2**(i+3),-1])
    pho = ncon((psi,psi.conj()),([-1,-3],[-2,-4]))
    pho = ncon((pho,mps[site-1],mps[site-1].conj()),([-1,-3,1,2],[-2,1],[-4,2]))
    pho = np.reshape(pho,(2**site,2**site))
    #site = len(mps)
    #pho = ncon((mps[0],mps[0]),([-1,-3],[-2,-4]))
    #for i in range(1, site-1):
    #    print(i)
    #    print(mps[i].shape)
    #    pho = ncon((pho,mps[i],mps[i]),([-1,-2,1,2],[1,-3,-5],[2,-4,-6]))
    #    pho = ncon((pho),([1,2,1,2,-1,-2]))
    #    print(pho.shape)
    #pho = ncon((pho,mps[site-1],mps[site-1]),([-1,-2,1,2],[-3,1],[-4,2]))
    #pho = ncon((pho),([1,2,1,2]))
    return pho


def mps_cFidelity(MPS, povm_M, Nqubit, S, logP):
    Fidelity = 0.0
    F2 = 0.0
    Ns = S.shape[0]
    L1 = 0.0
    L1_2 = 0.0
    for i in range(Ns):

        P = ncon((MPS[0].conj(), MPS[0], povm_M[S[i, 0], :, :]), ([1, -1], [2, -2], [1, 2]))

        # contracting the entire TN for each sample S[i,:]
        for j in range(1, Nqubit - 1):
            P = ncon((P, MPS[j].conj(), MPS[j], povm_M[S[i, j], :, :]), ([1, 2], [1, 3, -1], [2, 4, -2], [3, 4]))

        P = ncon((P, MPS[Nqubit - 1].conj(), MPS[Nqubit - 1], povm_M[S[i, Nqubit - 1], :, :]),
                 ([1, 2], [3, 1], [4, 2], [3, 4]))

        ratio = P / np.exp(logP[i])
        ee = np.sqrt(ratio)
        Fidelity = Fidelity + ee
        F2 = F2 + ee ** 2
        L1 = L1 + np.abs(1 - ratio)
        L1_2 = L1_2 + np.abs(1 - ratio)**2

    F2 = F2 / float(Ns)
    Fidelity = np.abs(Fidelity / float(Ns))
    Error = np.sqrt(np.abs(F2 - Fidelity ** 2) / float(Ns))
    L1_2 = L1_2 / float(Ns)
    L1 = np.abs(L1 / float(Ns))
    L1_err = np.sqrt(np.abs(L1_2 - L1 ** 2) / float(Ns))

    return np.real(Fidelity), np.real(Error), np.real(L1), np.real(L1_err)


def mps_cFidelity_model(MPS, povm_M, nb_qbits, samples_size, batch_size, model, device=torch.device("cpu")):

    sa, lp = model(forward_type="sample")
    sa = sa.cpu().numpy()
    lp = lp.cpu().numpy()
    cFid, cFidErr, KL, KLErr = mps_cFidelity(MPS, povm_M, nb_qbits, sa, lp)
    return cFid, cFidErr, KL, KLErr


def index(one_basis, base=4):
    """
    :return: integer as a string in base
    """
    return int(''.join(map(lambda x: str(int(x)), one_basis)), base)


def basis(num_sites, base=2):
  l_basis = []
  for i in range(base**num_sites):
    basis_str = np.base_repr(i, base=base, padding=num_sites)[-num_sites:]
    l_basis.append(np.array(list(basis_str), dtype=int))
    #l_basis.append(basis_str)
  l_basis = np.array(l_basis)
  return l_basis


def construct_ham(N, Jz=1.0, hx=1.0, boundary=0):
  # set Hamiltonian
  # boundary = 0 is OBC while boundary = 1 is PBC
  ham = np.zeros((2**N,2**N), dtype=complex)
  l_basis = basis(N, 2)
  for i in range(2**N):
    for j in range(N-1+boundary):
      ham[i, i] += - Jz *(4.0*l_basis[i, j] * l_basis[i, (j+1)% N] - 2.0*l_basis[i,j]- 2.0*l_basis[i,(j+1) % N] +1. )
      hop_basis = l_basis[i,:].copy()
      hop_basis[j] =  int(abs(1-hop_basis[j]))
      i_hop = index(hop_basis, 2)
      ham[i, i_hop] = -hx
    hop_basis = l_basis[i,:].copy()
    hop_basis[N-1] =  int(abs(1-hop_basis[N-1]))
    i_hop = index(hop_basis, 2)
    ham[i, i_hop] = -hx

  return ham

def ham_eigh(ham):
  w, v = np.linalg.eigh(ham)
  ind = np.argmin(w)
  E = w[ind]
  psi_g = v[:, ind]
  return psi_g, E


def kron_gate(gate, site, Nqubit, gate_factor=2):
    g = gate.copy()
    if site != 0:
      I_L = np.eye(2)
      for i in range(site-1):
        I_L = np.kron(I_L, np.eye(2))
    else:
      I_L = 1.

    if site != Nqubit - gate_factor:
      I_R = np.eye(2)
      for i in range(Nqubit-site-gate_factor-1):
        I_R = np.kron(I_R, np.eye(2))
    else:
      I_R = 1.

    g = np.kron(I_L, g)
    g = np.kron(g, I_R)

    return g


def Z1Z(site, Nqubit):
    """
    compute Z1Z correlation function
    """
    Z = np.array([[1, 0],[0, -1]])
    I_L = 1.0
    for i in range(site-1):
        I_L = np.kron(I_L, np.eye(2))

    I_R = 1.0
    for i in range(Nqubit-site-1):
        I_R = np.kron(I_R, np.eye(2))

    g = np.kron(Z, I_L)
    g = np.kron(g, Z)
    g = np.kron(g, I_R)

    return g


def ZZ(site1, site2, Nqubit):
    """
    compute Z1Z correlation function
    """
    Z = np.array([[1, 0],[0, -1]])
    I_L = 1.0
    I_M = 1.0
    I_R = 1.0
    for i in range(site1):
        I_L = np.kron(I_L, np.eye(2))
    for i in range(Nqubit-site2-1):
        I_R = np.kron(I_R, np.eye(2))
    for i in range(site2-site1-1):
        I_M = np.kron(I_M, np.eye(2))

    g = np.kron(I_L, Z)
    g = np.kron(g, I_M)
    g = np.kron(g, Z)
    g = np.kron(g, I_R)

    return g

def exp_X(beta, site, Nqubit):
    X = np.array([[0, 1],[1, 0]])
    Z = np.array([[1, 0],[0, -1]])
    gate = sp.linalg.expm(beta * X)
    return kron_gate(gate, site, Nqubit, gate_factor=1)

def exp_ZZ(gamma, site, Nqubit):
    X = np.array([[0, 1],[1, 0]])
    Z = np.array([[1, 0],[0, -1]])
    ZZ = gamma * np.kron(Z,Z)
    gate = sp.linalg.expm(ZZ)
    return kron_gate(gate, site, Nqubit)

def exp_ZZX(gamma, beta, site, Nqubit):
    X = np.array([[0, 1],[1, 0]])
    Z = np.array([[1, 0],[0, -1]])
    I = np.eye(2)
    XI = beta * (np.kron(X, I) + np.kron(I, X))
    ZZ = gamma * np.kron(Z,Z)
    expZZ = kron_gate(sp.linalg.expm(ZZ), site, Nqubit)
    expXI = kron_gate(sp.linalg.expm(XI), site, Nqubit)
    gate = expXI @ expZZ
    #print(np.linalg.det(expZZ))
    #print(np.linalg.det(expXI))
    #print(np.linalg.det(gate))
    return gate

def exp_ZZX_PBC(gamma, beta, Nqubit):
    X = np.array([[0, 1],[1, 0]])
    Z = np.array([[1, 0],[0, -1]])
    I_kron = np.eye(2)
    for i in range(Nqubit-3):
        I_kron = np.kron(I_kron, np.eye(2))
    ZZ = np.kron(Z,I_kron)
    ZZ = gamma * np.kron(ZZ,Z)
    X1 = np.kron(X,I_kron)
    X1 = np.kron(X1,np.eye(2))
    X2 = np.kron(np.eye(2), I_kron)
    X2 = np.kron(X2, X)
    XI = beta * (X1+X2)
    expZZ = sp.linalg.expm(ZZ)
    expXI = sp.linalg.expm(XI)
    gate = expXI @ expZZ
    return gate

def exp_ZZ_PBC(gamma, Nqubit):
    X = np.array([[0, 1],[1, 0]])
    Z = np.array([[1, 0],[0, -1]])
    #I_kron = np.eye(2)
    ZZ = Z.copy()
    for i in range(Nqubit-2):
        #I_kron = np.kron(I_kron, np.eye(2))
        ZZ = np.kron(ZZ, np.eye(2))
    #ZZ = np.kron(Z,I_kron)
    ZZ = gamma * np.kron(ZZ,Z)
    gate = sp.linalg.expm(ZZ)
    return gate


def Fidelity_test(Nqubit, target_vocab_size, povm, prob, pho, model, device=torch.device("cpu")):
    """
    test the fidelity between given prob and pho, and the model predicted prob and pho
    :return: cFid2 is classical fidelity and Fid2 is quantum fidelity
    """

    prob_povm = np.exp(vectorize(Nqubit, target_vocab_size, model, device))
    pho_povm = ncon((prob_povm,povm.Ntn),([1],[1,-1,-2]))
    #Et = np.trace(pho_povm @ povm.ham)
    #print('exact E:', E, 'current E:', Et.real)
    cFid = np.dot(np.sqrt(prob), np.sqrt(prob_povm))
    L1 = np.linalg.norm(prob-prob_povm,ord=1)
    KL = np.dot(prob, np.log(prob_povm))
    #Fid2 = ncon((pho,pho_povm),([1,2],[2,1])) ## true for 2 qubit
    Fid = np.abs(np.trace(sp.linalg.sqrtm(pho @ pho_povm @pho)))**2 ## true for pure state pho
    #print('cFid_ED: ', cFid2, Fid2)

    #a = np.array(list(it.product(range(4),repeat = Nqubit)))
    #a = torch.tensor(a)
    #l = torch.sum(torch.exp(model.logP(a)))
    #print("prob", l)
    return cFid, L1, KL, Fid


def Graph_test(step, mps, nb_qbits, samples_size, batch_size, model, device=torch.device("cpu")):
    gt = mps.Graph_t(step)
    sa, lp = model(forward_type="sample")
    sa = sa.cpu().numpy()
    lp = lp.cpu().numpy()
    cFid, std, L1, L1_err = mps.cFidelity_t(sa, lp, step)
    #print(cFid, std, L1, L1_err)
    return cFid, std, L1, L1_err



def target_state_gate(povm, circuit, step, mode="Graph"):
    """
    :return: povm_t and pho_t after applying gate
    """
    Nqubit = circuit.nb_qbits
    povm.construct_psi()
    povm.construct_Nframes()
    # GHZ state
    psi = np.zeros(2**Nqubit)
    psi[0] = 1.
    psi[-1] = 1.
    psi = psi/ np.sqrt(2)
    pho = np.outer(psi, np.conjugate(psi))
    prob = ncon((pho,povm.Mn),([1,2],[-1,2,1])).real

    # construct density matrix
    psi_t = povm.psi.copy()
    psi_t = psi_t / np.linalg.norm(psi_t)
    pho_t = np.outer(psi_t, np.conjugate(psi_t))
    prob_t = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real
    pho_t0 = pho_t.copy()
    prob_t0 = prob_t.copy()

    if mode == "Graph":
        SITE = []
        GATE = []
        P_GATE = []
        for i in range(step):
            SITE.append([i,i+1])
            GATE.append(povm.cz)
            P_GATE.append(povm.P_gate(povm.cz))
    elif mode == "GHZ":
        SITE = [[0]]
        GATE = [povm.H]
        P_GATE = [povm.P_gate(povm.H)]
        for i in range(step-1):
            SITE.append([i,i+1])
            GATE.append(povm.cnot)
            P_GATE.append(povm.P_gate(povm.cnot))
    else:
        assert False, "mode does not exist"
    for i in range(len(SITE)):
        sites=SITE[i]
        gate = P_GATE[i]
        gtype = int(gate.ndim/2)
        kron_gate = povm.kron_gate(GATE[i], sites[0], Nqubit)
        psi_t = psi_t @ kron_gate
        psi_t = psi_t / np.linalg.norm(psi_t)
        pho_t = np.outer(psi_t, np.conjugate(psi_t))
        prob_t = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real

    # test
    cFid2 = np.dot(np.sqrt(prob), np.sqrt(prob_t))
    ##cFid2 = np.linalg.norm(prob-prob_povm,ord=2)
    ##cFid2 = np.dot(prob, np.log(prob_povm))
    ##Fid2 = ncon((pho,pho_povm),([1,2],[2,1])) ## true for 2 qubit
    #Fid2 = np.square(np.trace(sp.linalg.sqrtm(pho_t0 @ pho_t @pho_t0))) ## true for pure state pho
    #print('Fidelity', cFid2, Fid2)
    print('Fidelity', cFid2)

    return prob_t, pho_t


def target_state_ham(povm, circuit, tau, step, boundary, mode='imag'):
    """
    :return: povm_t and pho_t after imaginary/real time evolution
    """
    Nqubit = circuit.nb_qbits
    povm.construct_psi()
    povm.construct_Nframes()
    povm.construct_ham(boundary)
    # prepare state
    #psi, E = povm.ham_eigh()
    #pho = np.outer(psi, np.conjugate(psi))
    #prob = ncon((pho,povm.Mn),([1,2],[-1,2,1])).real

    # construct density matrix
    psi_t = povm.psi.copy()
    psi_t = psi_t / np.linalg.norm(psi_t)
    pho_t = np.outer(psi_t, np.conjugate(psi_t))
    prob_t = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real
    pho_t0 = pho_t.copy()
    prob_t0 = prob_t.copy()

    # imaginary time
    if mode == 'imag':
        for i in range(step):
            pho_t = pho_t - tau *( povm.ham @ pho_t + pho_t @ povm.ham)
            #prob_t_raw = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real.astype(np.float32)
            pho_t = pho_t / np.trace(pho_t)
            prob_t = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real.astype(np.float32)
    elif mode == 'real':
        for i in range(step):
            pho_t = pho_t - 1j*tau *( povm.ham @ pho_t - pho_t @ povm.ham)
            #prob_t_raw = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real.astype(np.float32)
            pho_t = pho_t / np.trace(pho_t)
            prob_t = ncon((pho_t,povm.Mn),([1,2],[-1,2,1])).real.astype(np.float32)
            #print('Fid', np.linalg.norm(pho_t - pho_t0))
            #print('E', np.trace(pho_t @ povm.ham))
    else:
        assert False, 'mode does not exist'

    # test
    #cFid2 = np.dot(np.sqrt(prob_t0), np.sqrt(prob_t))
    ##cFid2 = np.linalg.norm(prob-prob_povm,ord=2)
    ##cFid2 = np.dot(prob, np.log(prob_povm))
    ##Fid2 = ncon((pho,pho_povm),([1,2],[2,1])) ## true for 2 qubit
    #Fid2 = np.square(np.trace(sp.linalg.sqrtm(pho_t0 @ pho_t @pho_t0))) ## true for pure state pho
    #print('Fidelity', cFid2, Fid2)

    return prob_t, pho_t


def compute_observables(obs, site, samples):
    """
    compute observables of an operator obs on site from given samp
    <O> = sum Pi obs_i ~ obs_i for samp
    :return: observable coefficient with respect to samples
    """
    ndim = obs.dim()
    if ndim == 1:
      Coef = obs[samples[:, site[0]]]
    elif ndim == 2:
      Coef = obs[samples[:, site[0]], samples[:, site[1]]]
    else:
      raise NameError("dimension not correct")

    return Coef.squeeze()


def compute_observables_correlation(obs, nb_qbits, num_batch, mini_batch_size, model):
    """
    compute observables of an operator obs on site from given samp
    <O> = sum Pi obs_i ~ obs_i for samp
    :return: observable coefficient with respect to samples
    """
    ndim = obs.dim()
    Ns = num_batch * mini_batch_size
    samp, _ = model(forward_type="sample")
    ob_matrix = torch.zeros(nb_qbits, nb_qbits, device=samp.device)
    ob2_matrix = torch.zeros(nb_qbits, nb_qbits, device=samp.device)
    for _ in range(num_batch):
      samp, _ = model(forward_type="sample")
      for i in range(nb_qbits):
          for j in range(i+1, nb_qbits):
              site = [i, j]
              Coef = compute_observables(obs, site, samp)
              Coef2 = Coef * Coef
              ob_matrix[i, j] += torch.mean(Coef)
              ob2_matrix[i, j] += torch.mean(Coef2)

    ob_matrix = ob_matrix / num_batch
    ob2_matrix = ob2_matrix / num_batch
    err_matrix = torch.sqrt((ob2_matrix - ob_matrix * ob_matrix) / Ns)
    ob_matrix = ob_matrix + torch.t(ob_matrix) + torch.eye(nb_qbits, device=samp.device)
    err_matrix = err_matrix + torch.t(err_matrix)

    return ob_matrix, err_matrix


def compute_energy(hl_ob, hlx_ob, Nqubit, samp):
    """
    compute expectation value of Hamiltonian, where Hamiltonian is TFIM formed from hlx_ob and hl_ob
    :return: mean of energy, standard deviation
    """
    Ns = samp.shape[0]
    Coef = compute_observables(hlx_ob, [Nqubit-2,Nqubit-1],samp)
    for i in range(Nqubit-2):
        Coef += compute_observables(hl_ob, [i,i+1], samp)
    Coef2 = Coef * Coef
    Coef_mean = torch.mean(Coef)
    Coef2_mean = torch.mean(Coef2)
    Err = torch.sqrt( (Coef2_mean- Coef_mean**2)/Ns)
    return Coef_mean, Err


def compute_energy_gpu(hl_ob, hlx_ob, Nqubit, num_batch, mini_batch_size, model):
    """
    compute expectation value of Hamiltonian on gpu, where Hamiltonian is TFIM formed from hlx_ob and hl_ob
    :return: mean of energy, standard deviation
    """
    Coef_mean = 0.0
    Coef2_mean = 0.0
    Ns = num_batch * mini_batch_size
    for _ in range(num_batch):
      samp, _ = model(forward_type="sample")
      Coef = compute_observables(hlx_ob, [Nqubit-2,Nqubit-1],samp)
      for i in range(Nqubit-2):
          Coef += compute_observables(hl_ob, [i,i+1], samp)
      Coef2 = Coef * Coef
      Coef_mean += torch.mean(Coef)
      Coef2_mean += torch.mean(Coef2)

    Coef_mean = Coef_mean / num_batch
    Coef2_mean = Coef2_mean / num_batch
    Err = torch.sqrt( (Coef2_mean- Coef_mean**2)/Ns)
    return Coef_mean, Err


def compute_energy_gpu_pbc(hl_ob, hlx_ob, Nqubit, num_batch, mini_batch_size, model):
    """
    compute expectation value of Hamiltonian on gpu, where Hamiltonian is TFIM formed from hlx_ob and hl_ob
    :return: mean of energy, standard deviation
    """
    Coef_mean = 0.0
    Coef2_mean = 0.0
    Ns = num_batch * mini_batch_size
    for _ in range(num_batch):
      samp, _ = model(forward_type="sample")
      Coef = compute_observables(hl_ob, [0,1], samp)
      for i in range(1,Nqubit):
          Coef += compute_observables(hl_ob, [i%Nqubit,(i+1)%Nqubit], samp)
      Coef2 = Coef * Coef
      Coef_mean += torch.mean(Coef)
      Coef2_mean += torch.mean(Coef2)

    Coef_mean = Coef_mean / num_batch
    Coef2_mean = Coef2_mean / num_batch
    Err = torch.sqrt( (Coef2_mean- Coef_mean**2)/Ns)
    return Coef_mean, Err


def compute_energy_mpo(Hp, S):
    E = 0.0;
    E2 = 0.0;
    N = len(Hp)
    Ns = S.shape[0]
    for i in range(Ns):

        # contracting the entire TN for each sample S[i,:]
        eT = Hp[0][S[i,0],:];

        for j in range(1,N-1):
            eT = ncon((eT,Hp[j][:,S[i,j],:]),([1],[1,-1]));

        j = N-1
        eT = ncon((eT,Hp[j][:,S[i,j]]),([1],[1]));
        #print i, eT
        E = E + eT;
        E2 = E2 + eT**2;
        Fest=E/float(i+1);
        F2est=E2/float(i+1);
        Error = np.sqrt( np.abs( F2est-Fest**2 )/float(i+1));
        #print i,np.real(Fest),Error
        #disp([i,i/Ns, real(Fest), real(Error)])
        #fflush(stdout);

    E2 = E2/float(Ns);
    E = np.abs(E/float(Ns));
    Error = np.sqrt( np.abs( E2-E**2 )/float(Ns));
    return np.real(E), Error


###############################################################


# TODO: What does this method do?
def flip3(samples, gate, k, site):
    """
    :param samples: Samples, nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm
    :param k: number of measurements? (for our case it is 4)
    :param site: the qbits affected by this gate
    :return:
    """
    nb_samples = samples.size(0)
    n_qbits = samples.size(1)
    # variable with the  flipped the measurements generated by a 2-qbit p_gate

    # repeat the samples K**2 sites. size = k**2 * nb_samples  x n_qbits
    flipped = samples.unsqueeze(1).repeat((1, k ** 2, 1)).view(k ** 2 * nb_samples, n_qbits)
    a = torch.tensor(list(product(range(k), repeat=2)))  # possible combinations of outcomes on 2 qubits

    # replacing the outcomes with the possible flipped outcomes
    flipped[:, site[0]] = a[:, 0].repeat(nb_samples)
    flipped[:, site[1]] = a[:, 1].repeat(nb_samples)

    # getting the coefficients of the p-gates that accompany the flipped samples
    coef = gate[:, :, samples[:, site[0]], samples[:, site[1]]]

    # TODO: Is this the right way to handle this error?
    if len(coef.shape) == 2:
        coef = coef.reshape(coef.shape[0], coef.shape[1], 1)

    # reshapes so that both coef and flipped have the same dim
    coef = np.reshape(np.transpose(coef[:, :, :], (2, 0, 1)), (nb_samples * k ** 2))
    coef = torch.tensor(coef.imag).float()

    # transform samples to one hot vector
    flipped = np.squeeze(np.reshape(np.eye(k)[flipped], [flipped.shape[0], n_qbits * k]).astype(np.uint8))
    flipped = torch.tensor(flipped).long()

    return flipped, coef


def flip2(samples, gate, k, site):
    """
    Given a sample state $a'$ and a 2qbit-gate, this method computes the associated states $a$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :return: A pair of tensors of sizes, (nb_samples * k**2, nb_qbits) and (nb_samples * k **2) respectively.
    """

    # TODO: when to specify device
    device = samples.device

    nb_samples = samples.size(0)
    nb_qbits = samples.size(1)
    # variable with the  flipped the measurements generated by a 2-qbit p_gate

    # repeat the samples K**2 sites. size = k**2 * nb_samples  x n_qbits
    flipped = samples.unsqueeze(1).repeat((1, k ** 2, 1)).view(k ** 2 * nb_samples, nb_qbits)
    a = torch.tensor(list(product(range(k), repeat=2)))  # possible combinations of outcomes on 2 qubits

    # replacing the outcomes with the possible flipped outcomes
    flipped[:, site[0]] = a[:, 0].repeat(nb_samples)
    flipped[:, site[1]] = a[:, 1].repeat(nb_samples)

    # getting the coefficients of the p-gates that accompany the flipped samples
    coef = gate[:, :, samples[:, site[0]], samples[:, site[1]]]

    # reshapes so that both coef and flipped have the same dim
    coef = coef.permute(2, 0, 1).contiguous().view(nb_samples * k ** 2)

    return flipped, coef


def flip2_reverse_presamples(samples, gate, k, site, model):
    """
    Given a sample state $a$ and a 2qbit-gate, this method computes the associated states $a'$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits) and (nb_samples) respectively.
    """

    # TODO: when to specify device
    device = samples.device

    nb_samples = samples.size(0)
    nb_qbits = samples.size(1)
    # variable with the  flipped the measurements generated by a 2-qbit p_gate

    # repeat the samples K**2 sites. size = k**2 * nb_samples  x n_qbits
    flipped = samples.unsqueeze(1).repeat((1, k ** 2, 1)).view(k ** 2 * nb_samples, nb_qbits)
    a = torch.tensor(list(product(range(k), repeat=2)))  # possible combinations of outcomes on 2 qubits

    # replacing the outcomes with the possible flipped outcomes
    flipped[:, site[0]] = a[:, 0].repeat(nb_samples)
    flipped[:, site[1]] = a[:, 1].repeat(nb_samples)

    # getting the coefficients of the p-gates that accompany the samples
    o_ab = gate[samples[:, site[0]], samples[:, site[1]], :, :]

    # reshapes so that both o_ab and flipped probability have the same dim
    o_ab = o_ab.view(nb_samples, k ** 2)

    # compute flipped probability
    pb = torch.exp(model.logP(flipped, look_ahead=True, device=device))
    pb = pb.view(nb_samples, k ** 2)
    # compute samples probability
    pa = torch.exp(model.logP(samples, look_ahead=True, device=device))
    # compute coef = sum {o_ab pb} / pa
    coef = torch.sum(o_ab * pb, dim=1) / pa


    return samples, coef


def flip2_reverse_core(samples, gate, k, site, model_copy):
    """
    Given a sample state $a$ and a 2qbit-gate, this method computes the associated states $a'$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :param model_copy: copy of model that does not take gradient and before update
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits) and (nb_samples) respectively.
    """

    device = samples.device

    nb_samples = samples.size(0)
    nb_qbits = samples.size(1)
    # variable with the  flipped the measurements generated by a 2-qbit p_gate

    # repeat the samples K**2 sites. size = k**2 * nb_samples  x n_qbits
    flipped = samples.unsqueeze(1).repeat((1, k ** 2, 1)).view(k ** 2 * nb_samples, nb_qbits)
    a = torch.tensor(list(product(range(k), repeat=2)))  # possible combinations of outcomes on 2 qubits

    # replacing the outcomes with the possible flipped outcomes
    flipped[:, site[0]] = a[:, 0].repeat(nb_samples)
    flipped[:, site[1]] = a[:, 1].repeat(nb_samples)

    # getting the coefficients of the p-gates that accompany the samples
    o_ab = gate[samples[:, site[0]], samples[:, site[1]], :, :]

    # reshapes so that both o_ab and flipped probability have the same dim
    o_ab = o_ab.view(nb_samples, k ** 2)

    # compute flipped probability
    pb = torch.exp(model_copy(forward_type="logP", seq=flipped))
    pb = pb.view(nb_samples, k ** 2)
    p_new = torch.sum(o_ab * pb, dim=1)

    return p_new


def flip2_reverse(samples, gate, k, site, model, model_copy):
    """
    Given a sample state $a$ and a 2qbit-gate, this method computes the associated states $a'$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :param model: model that takes gradient
    :param model_copy: copy of model that does not take gradient and before update
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits) and (nb_samples) respectively.
    """

    coef = flip2_reverse_core(samples, gate, k, site, model_copy)
    # compute samples probability
    device = samples.device
    pa = torch.exp(model.logP(samples, look_ahead=True, device=device))
    # compute coef = sum {o_ab pb} / pa
    coef = coef / pa

    return samples, coef


def reverse_samples_tfim(samples, logP_samples, hl, hlx, k, tau, model_copy):
    """
    Given a sample state $a$ and a 2qbit-gate, this method computes the associated states $a'$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param logP_samples: log probability of the Samples, nb_samples x 1
    :param hl: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of hl_com or hl_anti operator.
    :param hlx: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of hlx_com or hlx_anti operator.
    :param k: number of measurements? (for our case it is 4)
    :param tau: imaginary time evolution step
    :param model_copy: copy of model that does not take gradient and before update
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits) and (nb_samples) respectively.
    """

    device = samples.device
    nb_samples = samples.size(0)
    nb_qbits = samples.size(1)

    coef = flip2_reverse_core(samples, hlx, k, [nb_qbits-2, nb_qbits-1], model_copy)

    for i in range(nb_qbits-2):
        coef += flip2_reverse_core(samples, hl, k, [i, i+1], model_copy)

    # compute samples probability
    pb = torch.exp(logP_samples)
    p_new = pb - tau * coef

    return p_new


def reverse_samples_tfim_pbc(samples, logP_samples, hl, hlx, k, tau, model_copy):
    """
    Given a sample state $a$ and a 2qbit-gate, this method computes the associated states $a'$ such
    that $O_{aa'} \neq 0$ and the corresponding coefficients, note that there are only k**2 of these and the
    coefficients $O_{aa'}$ are given by the gate.
    :param samples: Samples, nb_samples x nb_qbits
    :param logP_samples: log probability of the Samples, nb_samples x 1
    :param hl: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of hl_com or hl_anti operator.
    :param hlx: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of hlx_com or hlx_anti operator.
    :param k: number of measurements? (for our case it is 4)
    :param tau: imaginary time evolution step
    :param model_copy: copy of model that does not take gradient and before update
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits) and (nb_samples) respectively.
    """

    device = samples.device
    nb_samples = samples.size(0)
    nb_qbits = int(samples.size(1))

    coef = flip2_reverse_core(samples, hl, k, [0, 1], model_copy)

    for i in range(1, nb_qbits):
        coef += flip2_reverse_core(samples, hl, k, [i%nb_qbits, (i+1)%nb_qbits], model_copy)

    # compute samples probability
    pb = torch.exp(logP_samples)
    p_new = pb - tau * coef

    return p_new


# TODO: in place method, deep copy of tensor
def sample_mcmc(samp, gate, k, site, epoch, model, model_copy):
    """
    Given a sample state $a$ and a 2qbit-gate, this method returns mcmc samples from updated probability after applying the gate.
    :param samp: Samples, nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :param epoch: number of epoch for mcmc
    :param model_copy: copy of model that does not take gradient and before update
    :return: A pair of tensors of sizes, (nb_samples, nb_qbits).
    """
    device = samp.device
    nb_samples = samp.size(0)
    nb_qbits = samp.size(1)
    ind = torch.randint(0, nb_qbits, (nb_samples,1))
    samples = samp

    for i in range(epoch*nb_qbits):
        mcmc_samples = torch.fmod(samples.scatter_add(1, ind, torch.ones((nb_samples,1), dtype=torch.long)), k)

        p_samples = flip2_reverse_core(samples, gate, k, site, model_copy)
        p_mcmc_samples = flip2_reverse_core(mcmc_samples, gate, k, site, model_copy)

        update_value = (p_mcmc_samples / p_samples > torch.rand(nb_samples)).long().unsqueeze(1)
        samples = torch.fmod(samples.scatter_add(1, ind, update_value), k)

    return samples


def sum_log_p(samples, model):
    """
    Returns the sum of the log probabilities.
    :param samples: Int tensor of size nb_samples x nb_qbits
    :param model: The model used to predict the likeness.
    :return: Sum of log probabilities.
    """

    # Get the current device that the samples are running on
    #model.eval()
    device = samples.device

    nb_measurements = 4
    nb_samples, nb_qbits = samples.size()

    # Given a sample a_0, ..., a_n changes it to SOS, a_0, ..., a_{n-1}
    init = nb_measurements * torch.ones((nb_samples, 1), dtype=torch.long, device=device)
    input_sample = torch.cat([init, samples], dim=1)[:, 0:nb_qbits]

    # The look_ahead mask is needed to only attend to previous qbits.
    probs = model(forward_type="normal", seq=input_sample, look_ahead=True)  # n_samples x seq_len x nb_measurements
    log_p = torch.log(torch.softmax(probs, dim=2) + 1e-10)

    # Creates the one_hot_encodding
    eye = torch.eye(nb_measurements).to(device)
    one_hot = eye[samples]

    log_p = (one_hot * log_p).sum(dim=1).sum(dim=1)
    #model.train()

    return log_p


def flip2_probs(samples, gate, k, site, model):
    """
    :param samples: Int tensor of size nb_samples x nb_qbits
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate, this gate should be the inverse gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :param model: The model used to predict the likeness.
    :return: sum O_ab p_NN_{b} of shape (nb_samples).
    """

    device = samples.device
    nb_samples = samples.size(0)
    nb_qbits = samples.size(1)
    nb_measurements = k

    # variable with the  flipped the measurements generated by a 2-qbit p_gate
    # repeat the samples K**2 sites. size = k**2 * nb_samples  x n_qbits
    flipped = samples.unsqueeze(1).repeat((1, k ** 2, 1)).view(k ** 2 * nb_samples, nb_qbits)
    a = torch.tensor(list(product(range(k), repeat=2)))  # possible combinations of outcomes on 2 qubits

    # replacing the outcomes with the possible flipped outcomes
    flipped[:, site[0]] = a[:, 0].repeat(nb_samples)
    flipped[:, site[1]] = a[:, 1].repeat(nb_samples)

    # getting the coefficients of the p-gates that accompany the samples
    o_ab = gate[samples[:, site[0]], samples[:, site[1]], :, :]
    # reshapes so that both o_ab and flipped probability have the same dim
    o_ab = o_ab.view(nb_samples, k ** 2)

    # Given a sample a_0, ..., a_n changes it to SOS, a_0, ..., a_{n-1}
    init = nb_measurements * torch.ones((nb_samples* k**2, 1), dtype=torch.long, device=device)
    input_sample = torch.cat([init, flipped], dim=1)[:, 0:nb_qbits]

    # The look_ahead mask is needed to only attend to previous qbits.
    probs = model(input_sample, look_ahead=True, device=device)  # n_samples x seq_len x nb_measurements
    log_p = torch.log(torch.softmax(probs, dim=2) + 1e-10)

    # Creates the one_hot_encodding
    eye = torch.eye(nb_measurements).to(device)
    one_hot = eye[flipped]

    pb = torch.exp((one_hot * log_p).sum(dim=1).sum(dim=1))
    pb = pb.view(nb_samples, k ** 2)
    p_new = torch.sum(o_ab * pb, dim=1)

    return p_new


def criterion(flipped, coef, gtype, model):
    """
    Returns the loss associated to
    :param flipped:
    :param coef: Coefficients ???
    :param gtype: Gate type.
    :param model: A model used to generate samples
    :return:
    """

    # Normalizing constant
    nb_measurements = model.config.nb_measurements
    f = nb_measurements if gtype == 1 else nb_measurements ** 2

    log_p = sum_log_p(flipped, model)

    return -f * (coef * log_p).mean()


def criterion2(samples, coef, model):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param coef: Coefficients ???
    :param model: A model to be optimized
    :return:
    """

    log_p = sum_log_p(samples, model)

    return -(coef * log_p).mean()


def supervised_loss(samples, model):
    """
    Returns the loss associated to
    :param samples: samples from exact sampling
    :param model: A model to be optimized
    :return:
    """

    log_p = sum_log_p(samples, model)

    return -log_p.mean()


def forward_KL(samples, p_new, logP_samples, model):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param p_new: updated probability
    :param logP_samples: log probability of the given samples
    :param model: A model to be optimized
    :return: sum (p_new / p_samples * log p_NN)
    """

    log_p = sum_log_p(samples, model)
    coef = p_new / torch.exp(logP_samples)
    coef = coef - coef.mean()

    return -(coef * log_p).mean()


def forward_L1(samples, p_new, logP_samples, model, gamma=0):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param p_new: updated probability
    :param logP_samples: log probability of the given samples
    :param model: A model to be optimized
    :param gamma: weighted by the power of p_samples
    :return: sum abs(p_new - p_NN) / p_samples
    """

    p_NN = torch.exp(sum_log_p(samples, model))
    p_samples = torch.exp(logP_samples)
    Loss = torch.abs(p_NN - p_new) * torch.pow(p_samples, gamma -1)
    #print(p_NN.requires_grad)
    #print(p_samples.requires_grad)
    #print(p_new.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()


def forward_L2(samples, p_new, logP_samples, model, gamma=0):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param p_new: updated probability
    :param logP_samples: log probability of the given samples
    :param gamma: weighted by the power of p_samples
    :param model: A model to be optimized
    :return: sum square(p_new - p_NN) / p_samples
    """

    p_NN = torch.exp(sum_log_p(samples, model))
    p_samples = torch.exp(logP_samples)
    Loss = torch.pow(p_NN - p_new, 2) * torch.pow(p_samples, gamma -1)
    #print(p_NN.requires_grad)
    #print(p_samples.requires_grad)
    #print(p_new.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()


def forward_cFid(samples, p_new, logP_samples, model, gamma=0):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param p_new: updated probability
    :param logP_samples: log probability of the given samples
    :param gamma: weighted by the power of p_samples
    :param model: A model to be optimized
    :return: sum sqaure(sqrt(abs(p_new)) - sqrt(p_NN)) / p_samples
    """

    p_NN = torch.exp(sum_log_p(samples, model))
    p_samples = torch.exp(logP_samples)
    Loss = torch.pow(torch.sqrt(p_NN) - torch.sqrt(torch.abs(p_new)), 2) * torch.pow(p_samples, gamma -1)
    #print(p_NN.requires_grad)
    #print(p_samples.requires_grad)
    #print(p_new.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()


def forward_cFid_KL(samples, p_new, logP_samples, model, gamma=0):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param p_new: updated probability
    :param logP_samples: log probability of the given samples
    :param gamma: weighted by the power of p_samples
    :param model: A model to be optimized
    :return: sum sqaure(sqrt(abs(p_new)) - sqrt(p_NN)) / p_samples
    """
    log_p = sum_log_p(samples, model)
    coef = 1.0 - torch.sqrt(torch.abs(p_new) / torch.exp(logP_samples))
    coef = coef - coef.mean()
    Loss = (coef * log_p).mean()

    #print(log_p.requires_grad)
    #print(p_new.requires_grad)
    #print(coef.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()


def forward_L2_KL(samples, p_new, logP_samples, model, gamma=0):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param p_new: updated probability
    :param logP_samples: log probability of the given samples
    :param gamma: weighted by the power of p_samples
    :param model: A model to be optimized
    :return: sum square(p_new - p_NN) / p_samples
    """

    log_p = sum_log_p(samples, model)
    p_samples = torch.exp(logP_samples)
    coef = 2.0 * (p_samples - p_new) * torch.pow(p_samples, gamma)
    coef = coef - coef.mean()
    Loss = (coef * log_p).mean()

    #print(log_p.requires_grad)
    #print(p_new.requires_grad)
    #print(coef.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()


def forward_L1_KL(samples, p_new, logP_samples, model, gamma=0):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param p_new: updated probability
    :param logP_samples: log probability of the given samples
    :param model: A model to be optimized
    :param gamma: weighted by the power of p_samples
    :return: sum abs(p_new - p_NN) / p_samples
    """

    log_p = sum_log_p(samples, model)
    p_samples = torch.exp(logP_samples)
    coef = torch.sign(p_samples - p_new) * torch.pow(p_samples, gamma)
    coef = coef - coef.mean()
    Loss = (coef * log_p).mean()

    #print(log_p.requires_grad)
    #print(p_new.requires_grad)
    #print(coef.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()


def forward_L1Nt_KL(samples, p_new, logP_samples, model, Nt_norm):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param p_new: updated probability
    :param logP_samples: log probability of the given samples
    :param model: A model to be optimized
    :param Nt_norm: norm of povm.Nt
    :return: sum abs(p_new - p_NN) / p_samples
    """

    log_p = sum_log_p(samples, model)
    p_samples = torch.exp(logP_samples)
    #print(samples.shape)
    #print(samples[:10])
    #print(Nt_norm)
    #print(torch.sign(p_samples).shape)
    #print(torch.prod(Nt_norm[samples], 1).shape)
    #print(torch.prod(Nt_norm[samples], 1)[:10])
    #print((3.7417**samples.shape[1]))
    coef = torch.sign(p_samples - p_new) * torch.prod(Nt_norm[samples], 1)
    coef = coef / (3.7417**samples.shape[1])
    coef = coef - coef.mean()
    Loss = (coef * log_p).mean()

    #print(log_p.requires_grad)
    #print(p_new.requires_grad)
    #print(coef.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()


def forward_L1zz_KL(samples, p_new, logP_samples, model, zz_ob):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param p_new: updated probability
    :param logP_samples: log probability of the given samples
    :param model: A model to be optimized
    :param zz_ob: povm observables
    :return: sum abs(p_new - p_NN) / p_samples
    """

    log_p = sum_log_p(samples, model)
    p_samples = torch.exp(logP_samples)
    nb_samples = samples.shape[0]
    nb_qbits = samples.shape[1]
    ob = torch.zeros(nb_samples, device=samples.device)
    for i in range(nb_qbits):
        for j in range(nb_qbits):
            ob += torch.abs(zz_ob[samples[:, i], samples[:, j]])

    coef = torch.sign(p_samples - p_new) * ob
    coef = coef - coef.mean()
    Loss = (coef * log_p).mean()

    #print(log_p.requires_grad)
    #print(p_new.requires_grad)
    #print(ob.requires_grad)
    #print(coef.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()


def reverse_L1(samples, logP_samples, gate, k, site, model, gamma=0):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate, this gate should be the inverse gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :param logP_samples: log probability of the given samples
    :param model: A model to be optimized
    :param gamma: weighted by the power of p_samples
    :return: sum abs(p_samples - p_NN) / p_samples
    """

    p_NN = flip2_probs(samples, gate, k, site, model)
    p_samples = torch.exp(logP_samples)
    Loss = torch.abs(p_NN - p_samples) * torch.pow(p_samples, gamma -1)
    #print(p_NN.requires_grad)
    #print(p_samples.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()


def reverse_L2(samples, logP_samples, gate, k, site, model, gamma=0):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate, this gate should be the inverse gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :param logP_samples: log probability of the given samples
    :param model: A model to be optimized
    :param gamma: weighted by the power of p_samples
    :return: sum square(p_samples - p_NN) / p_samples
    """

    p_NN = flip2_probs(samples, gate, k, site, model)
    p_samples = torch.exp(logP_samples)
    Loss = torch.pow(p_NN - p_samples, 2) * torch.pow(p_samples, gamma -1)
    #print(p_NN.requires_grad)
    #print(p_samples.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()


def reverse_cFid(samples, logP_samples, gate, k, site, model, gamma=0):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate, this gate should be the inverse gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :param logP_samples: log probability of the given samples
    :param model: A model to be optimized
    :param gamma: weighted by the power of p_samples
    :return: sum square(sqrt(p_samples) - sqrt(abs(p_NN))) / p_samples
    """

    p_NN = flip2_probs(samples, gate, k, site, model)
    p_samples = torch.exp(logP_samples)
    Loss = torch.pow(torch.sqrt(torch.abs(p_NN)) - torch.sqrt(p_samples), 2) * torch.pow(p_samples, gamma -1)
    #print(p_NN.requires_grad)
    #print(p_samples.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()


def reverse_KL(samples, logP_samples, gate, k, site, model):
    """
    Returns the loss associated to
    :param samples: samples from important sampling
    :param gate: this is a 4x4x4x4 np array obtained from povm, which corresponds to the probability representation of
        the gate, this gate should be the inverse gate.
    :param k: number of measurements? (for our case it is 4)
    :param site: The position of the two qbits affected by this gate
    :param logP_samples: log probability of the given samples
    :param model: A model to be optimized
    :param gamma: weighted by the power of p_samples
    :return: sum - p_samples log(abs(p_NN))
    """

    p_NN = flip2_probs(samples, gate, k, site, model)
    p_samples = torch.exp(logP_samples)
    Loss = - p_samples * torch.log(torch.abs(p_NN) + 1e-13)
    #print(p_NN.requires_grad)
    #print(p_samples.requires_grad)
    #print(Loss.requires_grad)
    #assert False, 'stop'

    return Loss.mean()
