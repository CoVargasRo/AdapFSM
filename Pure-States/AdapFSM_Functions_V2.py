import numpy as np
from scipy.optimize import minimize

# Construction of the Fisher Symmetric Measurements
def FSM( d, phase=1 ): #with phase=1, it builds the negative povm

    n = 2*d - 1 # number of measurement elements

    psi_k = np.zeros( (d, n), dtype=complex ) # array of the povm elements

    # first element of the povm
    psi_0 = np.ones(d, dtype=complex) 
    psi_0[1:] = - phase* np.exp( 1j*np.pi/4 )
    psi_0 = psi_0 / np.sqrt( n )

    psi_k[:,0] = psi_0

    # the other elements of the povm
    for k in range(1,d):
        psi = np.ones(d, dtype=complex) 
        psi[1:] = - phase*np.exp( 1j*np.pi/4 ) / ( np.sqrt(n) + 1 )
        psi[k] += phase*np.sqrt(0.5*n)
        psi = psi / np.sqrt(n)

        psi_k[:,2*k-1] = psi 

        psi = np.ones(d, dtype=complex) 
        psi[1:] = - phase*np.exp( 1j*np.pi/4 ) / ( np.sqrt(n) + 1 )
        psi[k] += phase*1j*np.sqrt(0.5*n)
        psi = psi / np.sqrt(n)

        psi_k[:,2*k] = psi 

    return psi_k

# Random state builder
def RandomState( d ): 
    psi = np.random.randn( d ) + 1j*np.random.randn( d )
    psi = psi / np.linalg.norm( psi )
    return psi

# Simulator of the measurement outcomes
def SimMeas( state, measure, shots, return_counts=False):

    probs = np.abs( measure.T.conj() @ state )**2
    probs = np.clip(probs, 0, None)   # removes negative values ​​resulting from rounding
    probs = probs / probs.sum()       # renormalize so that they add up to exactly 1

    shots = int( round( shots ) )
    counts = np.random.multinomial( shots, probs )
    return counts.astype(float) if return_counts else counts / shots

# Fidelity between two pure states
def Fidelity( psi, phi ):  
    return np.abs(np.vdot( psi, phi ) )**2

# Reordering between complex and real vectors
def complex2real( A ):
    return np.concatenate( ( np.real(A), np.imag(A) ), axis=0 )    

def real2compelex( A ):
    d  = len(A)//2 
    return A[:d] + 1j*A[d:] 

# It builds the basis with first element the fiducial one
def BasisWithFiducial(fid):
    """A dxd unitary matrix whose first column is the fiducial state (normalized)."""
    fid = fid / np.linalg.norm(fid)
    d = len(fid)
    j = np.argmax(np.abs(fid))                      
    cols = [fid] + [np.eye(d, dtype=complex)[:, k] for k in range(d) if k != j]
    Q, R = np.linalg.qr(np.column_stack(cols))
    Q[:, 0] = fid                                   
    return Q

#Maximum Likelihood Estimation for pure states

def MLE_pure( d, probs_ex, 
                measures, 
                init_state=None,
                n_random_starts=0 ):  

    wtot = probs_ex.sum()          # W = sum_alpha w_alpha

    
    def likelihood( parametros ): 
        psi = real2compelex( parametros )
        psi = psi / np.linalg.norm( psi ) 
        probs_th = np.abs( measures.T.conj() @ psi ) ** 2 + 1e-300 #The term avoids divergence at ln(0) = −infty
        return -np.sum(np.log(probs_th) * probs_ex) / wtot
    
    beta_k = measures.real
    gamma_k = measures.imag

    def DerProbs(parametros):
        """"
        Derivatives of the probabilities with respect to the real and imaginary parameters of the state.
        """

        d = len(parametros)//2
        a = parametros[:d]
        b = parametros[d:]

        # Normalization of the state
        N = np.sqrt(np.sum(a**2 + b**2))
        a_tilde = a / N
        b_tilde = b / N

        # Theoretical probabilities
        meas  = beta_k - 1j*gamma_k
        state = a + 1j*b
        state = state / N
        probs_th = np.abs( meas.T @ state)**2

        # Derivatives of the probabilities with respect to the real and imaginary parameters of the state

        suma1_alpha = beta_k.T @ a_tilde   + gamma_k.T @ b_tilde   
        suma2_alpha = gamma_k.T @ a_tilde  - beta_k.T @ b_tilde  

        der_ak = 2/N*(beta_k * suma1_alpha + gamma_k * suma2_alpha) - 2*a[:, np.newaxis] /N**2 * probs_th[np.newaxis, :]
        der_bk = 2/N*(gamma_k * suma1_alpha - beta_k * suma2_alpha) - 2*b[:, np.newaxis] /N**2 * probs_th[np.newaxis, :]

        return np.vstack((der_ak, der_bk)).T

    def MLE_jac(parametros):
        """"
        Gradient of the likelihood function.
        """
        # Pure state corresponding to the real parameters
        psi = real2compelex( parametros) 
        psi = psi/np.linalg.norm(psi) #normalization of the state

        # Theoretical probabilities of the measurements with the current estimated state
        probs_th = np.abs( measures.T.conj() @ psi)**2 + 1e-300

        der = DerProbs(parametros)

        return -np.sum((probs_ex / probs_th) * der.T , axis = 1) / wtot

    # Initial points for the optimization

    starts = [] if init_state is None else [init_state] # if an initial state is provided, it is considered as a starting point for the optimization
    starts += [ RandomState(d) for _ in range(n_random_starts) ] # consider a random selection of n random states apart from the provided initial state
    if not starts:
        starts = [ RandomState(d) ] # if we don't want any random starting points, only consider one

    best = None
    for s in starts:
        res = minimize( likelihood, complex2real( s / np.linalg.norm(s) ),
                        method='BFGS', jac=MLE_jac )
        if best is None or res.fun < best.fun:
            best = res

    state_est = real2compelex( best.x )
    return state_est / np.linalg.norm( state_est )

# Adaptive FSM protocol with MLE

def AdapFSM_V2(state , ensamble , n_random_starts=0):
    """
    Adaptive FSM protocol with MLE

    Parameters:
        state : unknown pure state to be estimated
        ensamble : number of copies of the state available
        n_random_starts: number of random starting points for the MLE optimization

    Return:
        Fidelities after first estimation and final estimation

    """

    d = len(state)

    # Number of shots for the first and second measurements
    N1 = round(((d-1)*(2*d-1)/4)**(1/3) * ensamble**(2/3))/2
    N2 = ensamble - 2*N1 - 10*d

   
    #Step 0: measure the state in the computational basis to choose a fiducial state
    #Choose the fiducial state as the one with the highest probability in the computational basis

    BaseComp = np.eye(d, dtype=complex)
    meas  = SimMeas(state, BaseComp, shots=10*d)
    index = np.argmax(meas)

    order = [index] + [j for j in range(d) if j != index]
    Base  = np.eye(d, dtype=complex)[:, order]     # columna 0 = e_index


    #Step 1: build the FSM, measure them, find coefficients and phases, first estimation

    # Positive FSM
    fsm_plus = FSM( d , -1 )
    fsm_plus_1 = Base@fsm_plus

    # Negative FSM
    fsm_minus = FSM( d )
    fsm_minus_1 = Base@fsm_minus

    measures_accumulated = np.concatenate([BaseComp, fsm_plus_1, fsm_minus_1], axis=1)

    # Simulate the measurements of the state with the FSMs
    probs_plus  = SimMeas(state, fsm_plus_1, N1)
    probs_minus = SimMeas(state, fsm_minus_1, N1)

    w_acc = np.concatenate([10*d*meas ,N1*probs_plus, N1*probs_minus]) #weights accumulated


    # Coefficients of the FSM in the fiducial basis
    b0 = fsm_plus[0].real
    bk = fsm_plus[1:].real
    ck = fsm_plus[1:].imag

    Delta_k = np.sum( ( (bk + 1j*ck)/b0 ) * ( probs_plus - probs_minus ), axis=1 )  

    # a0^2 estimated as the median of the ratios (|Delta_k|^2 - b0^2*sum(|Delta_k|^2)) / (2*(probs_plus + probs_minus) - 4*b0^2)

    num = np.abs((bk - 1j*ck).T @ Delta_k)**2 - b0**2*np.sum(np.abs(Delta_k)**2)
    den = 2*(probs_plus + probs_minus) - 4*b0**2

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num/den

    ok = np.isfinite(ratio) & (ratio > 0) & (ratio < 1)
    mean_a0 = np.sqrt(np.median(ratio[ok])) if ok.any() else 1/np.sqrt(d)
 
    state_0 = np.hstack((mean_a0, Delta_k/(2*mean_a0)))
    state_hat = Base @ (state_0/np.linalg.norm(state_0))

    #Utilize MLE to refine the estimation

    state_hat_est = MLE_pure( d, w_acc, measures_accumulated, state_hat, n_random_starts ) #agregar datos anteriores
    Fid_1 = Fidelity(state_hat_est, state)
    
    #Step 2: create FSM with previous estimation, measure and reconstruct with MLE

    # Adapted FSM with the previous estimation as fiducial state
    Base = BasisWithFiducial(state_hat_est)
    fsm  = Base @ fsm_minus
    measures_accumulated_all = np.concatenate([measures_accumulated,fsm], axis=1)

    # Simulate the measurements of the state with the adapted FSM
    probs_3 = SimMeas( state, fsm , N2)

    w_acc_all = np.concatenate([w_acc , N2*probs_3])

    # Final estimation with MLE using all the accumulated data
    state_est = MLE_pure( d, w_acc_all, measures_accumulated_all, state_hat_est , n_random_starts) #agregar datos anteriores

    Fid_2 = Fidelity(state_est, state)
    
    return Fid_1, Fid_2

    


