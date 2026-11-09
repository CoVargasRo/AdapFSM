import numpy as np
from scipy.optimize import minimize

# Construction of the Fisher Symmetric Measurements
def FSM( d, phase=1 ): #with phase=1, the negative povm is constructed

    n = 2*d - 1
    psi_k = np.zeros( (d, n), dtype=complex )

    psi_0 = np.ones(d, dtype=complex) 
    psi_0[1:] = - phase* np.exp( 1j*np.pi/4 )
    psi_0 = psi_0 / np.sqrt( n )

    psi_k[:,0] = psi_0

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

# Random pure state
def RandomState( d ): 
    psi = np.random.randn( d ) + 1j*np.random.randn( d )
    psi = psi / np.linalg.norm( psi )
    return psi

# Exact measurements for mixed state
def Meas_th(rho , measure):

    n = len(measure[0])  # amount of measure elements

    probs = np.zeros(n)

    for i in range(n):
        probs[i] = np.abs(measure[:,i].conj().T @ rho @ measure[:,i])

    return probs

# Simulated measurements for mixed state
def SimMeas( rho , measure, shots, return_counts=False):

    n = len(measure[0])  # amount of measure elements

    probs = np.zeros(n)

    for i in range(n):
        probs[i] = np.abs(measure[:,i].conj().T @ rho @ measure[:,i])

    probs = np.clip(probs, 0, None)   # elimina valores negativos por redondeo
    probs = probs / probs.sum()       # renormaliza para que sumen exactamente 1
    
    shots = round( shots )
    counts = np.random.multinomial( shots, probs )

    return  counts.astype(float) if return_counts else counts / shots

# Fidelity between pure states
def FidelityPure( psi, phi ):  
    return np.abs(np.vdot( psi, phi ) )**2

#############################################################################################
def complex2real( A ):
    return np.concatenate( ( np.real(A), np.imag(A) ), axis=0 )    

def real2compelex( A ):
    d  = len(A)//2 
    return A[:d] + 1j*A[d:] 


#Maximum Likelihood Estimation for sligthly mixed states

def MLE_mixed( d , 
              probs_exp , 
              measures ,
              init_state=None, 
              lamda = None, 
              n_random_starts=0 ):

    def sigma(t):
        return 1/(1+np.exp(-t))

    # coefficients of the measurement elements FS
    beta_k = measures.real
    gamma_k = measures.imag

    varphi = np.sum(beta_k**2 + gamma_k**2, axis=0).real

    w_tot = probs_exp.sum() 

    # Likelihood function

    def likelihood( parametros ): 
            psi = real2compelex( parametros[:-1] )
            psi = psi / np.linalg.norm( psi ) 
            t = parametros[-1]
    
            probs_th = (1-sigma(t))*np.abs(measures.T.conj()@psi)**2 + (sigma(t)/d)*varphi + 1e-300 #el sumando evita la divergencia en ln(0)=-infty
            
            return -np.sum(np.log(probs_th) * probs_exp)/w_tot
    
    beta_k = measures.real
    gamma_k = measures.imag

    def DerProbs(parametros):
        """"
        Derivatives of probabilities with respect to real and imaginary parameters of the state.
        """

        d = len(parametros)//2
        a = parametros[:d]
        b = parametros[d:-1]
        t = parametros[-1]

        lamda = sigma(t)

        # normalize the state
        N = np.sqrt(np.sum(a**2 + b**2))
        a_tilde = a / N
        b_tilde = b / N

        # theorical probabilities
        meas  = beta_k - 1j*gamma_k
        state = a + 1j*b
        state = state / N
        probs_th = np.abs( meas.T @ state)**2

        # partial derivatives
        suma1_alpha = beta_k.T @ a_tilde   + gamma_k.T @ b_tilde  
        suma2_alpha = gamma_k.T @ a_tilde  - beta_k.T @ b_tilde 

        der_ak = 2/N*(beta_k * suma1_alpha + gamma_k * suma2_alpha) - 2*a[:, np.newaxis] /N**2 * probs_th[np.newaxis, :]
        der_bk = 2/N*(gamma_k * suma1_alpha - beta_k * suma2_alpha) - 2*b[:, np.newaxis] /N**2 * probs_th[np.newaxis, :]

        return (1-lamda)*np.vstack((der_ak, der_bk)).T
    
    def Derivada_lamda(parametros):
        # Calculate the derivative of the probability of each measurement element with respect to the parameter a_k
        
        t = parametros[-1]
        lamda = sigma(t)
        d  = len(parametros)//2 
        ak = parametros[:d] 
        bk = parametros[d:-1] 

        u = ak + 1j*bk
        u = u / np.linalg.norm(u)

        sum1 = np.sum((beta_k - 1j*gamma_k).T * u, axis=1)

        return lamda*(1-lamda)*(-np.abs(sum1)**2 + varphi/d)

    def MLE_jac(parametros):
        """"
        Likelihood function gradient.
        """
        state = real2compelex(parametros[:-1])
        psi = state / np.linalg.norm(state)
        t = parametros[-1]

        probs_th = (1-sigma(t))*np.abs(measures.T.conj()@psi)**2 + (sigma(t)/d)*varphi + 1e-300 #el sumando evita la divergencia en ln(0)=-infty

        d_ab = DerProbs(parametros).T
        d_lamda = Derivada_lamda(parametros)

        der = np.vstack([d_ab , d_lamda])

        return -np.sum((probs_exp / probs_th) * der , axis = 1)/w_tot

  
    ## Likelihood minimization
    if init_state is None:
        init_state = RandomState(d)
    if lamda is None:
        lamda = np.abs(np.random.normal(0, 0.001))

    lamda = float(np.clip(lamda, 1e-8, 1 - 1e-8))
    t = np.log(lamda/(1 - lamda))

    def make_start(s):
        return np.hstack((complex2real(s/np.linalg.norm(s)), t))

    starts  = [make_start(init_state)]
    starts += [make_start(RandomState(d)) for _ in range(n_random_starts)]

    best = None
    for s in starts:
        res = minimize(likelihood, s, method='BFGS', jac=MLE_jac)
        if best is None or res.fun < best.fun:
            best = res

    state_est = real2compelex(best.x[:-1])
    state_est = state_est/np.linalg.norm(state_est)
    lamda = sigma(best.x[-1]).real

    return state_est, lamda

def BasisWithFiducial(fid):
    """Matriz unitaria dxd cuya primera columna es fid (normalizado)."""
    fid = np.asarray(fid, dtype=complex)
    fid = fid / np.linalg.norm(fid)
    d = len(fid)
    j = np.argmax(np.abs(fid))                       # componente de mayor módulo
    cols = [fid] + [np.eye(d, dtype=complex)[:, k] for k in range(d) if k != j]
    Q, R = np.linalg.qr(np.column_stack(cols))
    Q[:, 0] = fid                                    # fija signo/fase de la 1ª columna
    return Q

## AdapFSM protocol for mixed states

def AdapFSM_mixed_V2(psi , lamda ,ensamble, n_random_starts=0):

    d = int(len(psi))
    rho = (1-lamda)*np.outer(psi, np.conj(psi)) + (lamda/d)*np.eye(d, dtype=complex)
    
    # division de ensamble

    N1 = 10*d
    N2 = round(((d-1)*(2*d-1)/4)**(1/3) * ensamble**(2/3))/2
    N3  = int(ensamble - N1 - 2*N2) 
    
    assert N3 > 0, "ensamble demasiado pequeño para esta división"

    #STEP 0: Measure computational basis and define fiducial state
    BaseComp = np.eye(d, dtype=complex)
    meas_raw = SimMeas(rho, BaseComp, N1)
    index = int(np.argmax(meas_raw)) # indice del resultado más frecuente
    perm = np.arange(d); perm[[0, index]] = perm[[index, 0]]
    Base = np.eye(d, dtype=complex)[:, perm] #base reordenada
    meas = meas_raw[perm] # frecuencias reordenadas, meas[0]= frecuencia del fiducial

    
    #####################################################################################
    #STEP 1: Build FSMs, measure them, find coefficients and phases, first estimate
    fsm_plus = FSM( d , -1 )
    fsm_plus_1 = Base@fsm_plus

    fsm_minus = FSM( d )
    fsm_minus_1 = Base@fsm_minus

    measures_accumulated = np.concatenate([fsm_plus_1, fsm_minus_1, BaseComp], axis=1)

    #Simulated measurements
    probs_plus  = SimMeas(rho, fsm_plus_1 , N2)
    probs_minus = SimMeas(rho, fsm_minus_1, N2)

    probs_accumulated = np.concatenate([probs_plus*N2, probs_minus*N2, meas_raw ], axis=0)

    #Coefficients of the measuring elements
    b0 = fsm_plus[0].real
    bk = fsm_plus[1:].real
    ck = fsm_plus[1:].imag

    varphi = (b0**2 + np.sum( bk**2 + ck**2 , axis=0)).real

    #Calculation of delta_k
    Delta_k = np.sum( ( (bk + 1j*ck)/b0 ) * ( probs_plus - probs_minus ), axis=1 )  

    #Calculation of lambda
    lamda_k = (d/2) * ((meas[0] + meas[1:] - np.sqrt((meas[0] - meas[1:])**2 + np.abs(Delta_k**2))))

    lamda_est = float(np.clip(np.mean(lamda_k), 1e-8, 0.5)) # considera un lambda inicial entre 1e-8 y 0.5 para evitar problemas de convergencia

    #Equation for a0^2
    num = np.abs((bk - 1j*ck).T @ Delta_k)**2 - b0**2*np.sum(np.abs(Delta_k)**2)
    den = 2*(1-lamda_est)*(probs_plus + probs_minus) \
        - 4*(1-lamda_est)*(lamda_est/d)*varphi - 4*(1-lamda_est)**2*b0**2
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num/den
    a0 = np.isfinite(ratio) & (ratio > 0) & (ratio < 1)
    mean_a0 = np.sqrt(np.median(ratio[a0])) if a0.any() else 1/np.sqrt(d)

    # Estimation of the ak coefficients
    ak = Delta_k / (2*(1-lamda_est)*mean_a0)

    state_0 = np.hstack( ( mean_a0 , ak))
    state_hat = Base @ ( state_0 / np.linalg.norm( state_0 ) )

    #Fid0 = FidelityPure(psi , state_hat)  #Fidelidad entre el estado estimado y el real

    #STEP 2: Use MLE to refine the estimate

    state_hat_est , lamda_hat_est = MLE_mixed( d , probs_accumulated, measures_accumulated, state_hat , lamda_est , n_random_starts) #add the previous data

    Fid1 = FidelityPure(state_hat_est, psi)  #Accuracy between the estimated and actual state

    #STEP 3: Create FSM with previous estimate, measure and estimate with MLE

    Base = BasisWithFiducial(state_hat_est)
    fsm = Base @ fsm_minus

    ## Simulated measurements
    probs_3 = SimMeas( rho, fsm, N3)
    measures_all = [fsm_plus_1, fsm_minus_1, fsm, BaseComp]
    probs_all    = [probs_plus*N2, probs_minus*N2,
                    probs_3*N3,      meas_raw]

    psi_est, lamda_hat = MLE_mixed(d, np.concatenate(probs_all),
                                np.concatenate(measures_all, axis=1),
                                state_hat_est, lamda_hat_est, n_random_starts)

    Fid2 = FidelityPure(psi_est, psi)

    return Fid1 , Fid2