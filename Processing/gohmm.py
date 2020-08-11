import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

try:
    from numba import njit
except ModuleNotFoundError:
    njit = lambda f: f

@njit
def logsumexp(x):
    mx = np.max(x)
    return np.log(np.sum(np.exp(x - mx))) + mx

@njit
def softmax(x):
    return np.exp(x - logsumexp(x))

def forward_backward_old(transition_probs, observations, initial_probs=None):
    observations = np.array(list(observations))
    N = len(transition_probs)
    T = len(observations)
    if initial_probs is None:
        initial_probs = np.ones(N)
        initial_probs /= np.sum(initial_probs)
    
    forward_probs = np.zeros((T, N))
    backward_probs = forward_probs.copy()

    forward_probs[0] = initial_probs*observations[0]
    for i in range(1, T):
        forward_probs[i] = forward_probs[i-1]@transition_probs
        forward_probs[i] *= observations[i]
    
    backward_probs[-1] = 1.0
    backward_probs[-1] /= np.sum(backward_probs[-1])
    for i in range(T-2, -1, -1):
        backward_probs[i] = observations[i+1]*backward_probs[i+1]@transition_probs.T
    
    state_probs = forward_probs*backward_probs
    state_probs /= np.sum(state_probs, axis=1).reshape(-1, 1)
    return state_probs, np.log(forward_probs), np.log(backward_probs)

@njit
def forward_backward(transition_probs, observations, initial_probs):
    observations = np.log(observations)
    N = len(transition_probs)
    T = len(observations)
    if initial_probs is None:
        initial_probs = np.ones(N)
        initial_probs /= np.sum(initial_probs)
    
    forward_probs = np.zeros((T, N))
    backward_probs = forward_probs.copy()
    
    transition_probs = np.log(transition_probs)

    forward_probs[0] = np.log(initial_probs) + observations[0]
    tmp = np.zeros(N)
    for t in range(1, T):
        for till in range(N):
            for fron in range(N):
                tmp[fron] = forward_probs[t-1,fron] + transition_probs[fron,till]
            forward_probs[t,till] = logsumexp(tmp)
        forward_probs[t] += observations[t]
    
    backward_probs[-1] = np.log(1.0)
    for t in range(T-2, -1, -1):
        for fron in range(N):
            for till in range(N):
                tmp[till] = observations[t+1,till] + backward_probs[t+1,till] + transition_probs[fron,till]
            backward_probs[t,fron] = logsumexp(tmp)
    
    state_probs = forward_probs + backward_probs
    #state_probs /= np.sum(state_probs, axis=1).reshape(-1, 1)
    for t in range(T):
        state_probs[t] = softmax(state_probs[t])
    return state_probs, forward_probs, backward_probs

@njit
def compute_log_xi(emission_probs, transition, forward, backward):
    N = len(transition)
    T = len(forward)
    xi = np.zeros((T, N, N))
    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                xi[t,i,j] = forward[t,i] + np.log(transition[i,j]) + backward[t+1,j] + np.log(emission_probs[t+1,j])
    
    return xi

@njit
def compute_log_xi_sum(emission_probs, transition, forward, backward):
    # Not really sure how or why this works. The logic is copied from hmmlearn.
    # A more readable version is in compute_transition_matrix
    N = len(transition)
    T = len(forward)
    xi = np.full((N, N), -np.inf)
    total = logsumexp(forward[-1])
    tmp = xi.copy()
    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                tmp[i,j] = forward[t,i] + np.log(transition[i,j]) + backward[t+1,j] + np.log(emission_probs[t+1,j]) - total
        
        for i in range(N):
            for j in range(N):
                xi[i,j] = np.logaddexp(xi[i,j], tmp[i, j])
    return xi

@njit
def compute_transition_matrix(emission_probs, transition, state_prob, forward, backward):
    N = len(transition)
    T = len(forward)
    xi = np.zeros((T, N, N))

    xi = compute_log_xi(emission_probs, transition, forward, backward)
    a = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            a[i,j] = logsumexp(xi[:-1,i,j])
            #a[i,j] -= logsumexp(forward[:-1,i] + backward[:-1,i])
    
    a = np.exp(a)
    a /= np.sum(a, axis=1).reshape(-1, 1)
    transition_est = a
    initial_est = state_prob[0]

    return transition_est, initial_est

def compute_gaussian_mean(emissions, state_prob):
    state_weights = state_prob/np.sum(state_prob, axis=0)
    means_est = np.sum(emissions.reshape(-1, 1)*state_weights, axis=0)
    return means_est

def compute_gaussian_std(means, emissions, state_prob):
    state_weights = state_prob/np.sum(state_prob, axis=0)
    diffs = emissions.reshape(-1, 1) - means
    vars_est = np.sum((diffs**2)*(state_weights), axis=0)
    stds_est = np.sqrt(vars_est)
    return stds_est

def compute_gaussian_emission_distribution(emissions, state_prob):
    means_est = compute_gaussian_mean(emissions, state_prob)
    stds_est = compute_gaussian_std(means_est, emissions, state_prob)

    return means_est, stds_est

def simulate_process(T, initial_probs, state_transitions, emission_dists):
    Nstates = len(initial_probs)
    states = []
    emissions = []
    state = np.random.choice(Nstates, p=initial_probs)
    for i in range(T):
        emission = emission_dists[state].rvs()
        states.append(state)
        emissions.append(emission)
        state = np.random.choice(Nstates, p=state_transitions[state])

    return np.array(states), np.array(emissions)

def get_demo_process():
    means = np.array([-1.0, 1.0, 2.0])
    stds = np.repeat(1.0, len(means))*0.5

    Nstates = len(means)
    
    trans = np.random.rand(Nstates, Nstates)
    trans += np.eye(*trans.shape)
    initial = np.random.rand(Nstates)
    initial /= np.sum(initial)
    trans /= np.sum(trans, axis=1).reshape(-1, 1)
    
    return initial, trans, (means, stds)

def normalize1(x):
    x /= np.sum(x, axis=1).reshape(-1, 1)
    return x

def verify_reest_multi():
    np.random.seed(0)
    initial, trans, (means, stds) = get_demo_process()

    Nstates = len(trans)
    
    initial_guess = np.ones(Nstates)
    initial_guess /= np.sum(initial_guess)

    transition_guess = np.ones((Nstates, Nstates))
    transition_guess /= np.sum(transition_guess, axis=1).reshape(-1, 1)
    
    means_guess = means + np.random.randn(3)
    stds_guess = stds

    N_trials = 100
    T = 100
    
    emission_dists = [scipy.stats.norm(m, s) for m, s in zip(means, stds)]

    all_emissions = [simulate_process(T, initial, trans, emission_dists)[1] for i in range(N_trials)]
    
    transition_est = transition_guess
    initial_est = initial_guess
    means_est = means_guess
    stds_est = stds_guess
    niter = 100

    for itr in range(niter):
        emission_ests = [scipy.stats.norm(m, s) for m, s in zip(means_est, stds_est)]
        state_probs = []
        total_lik = 0
        trans = np.zeros((Nstates, Nstates))
        for emissions in all_emissions:
            emission_probs = np.array([d.pdf(emissions) for d in emission_ests]).T
            state_prob, forward, backward = forward_backward(transition_est, emission_probs, initial_est)
            state_probs.append(state_prob)
            wtf = compute_log_xi_sum(emission_probs, transition_est, forward, backward)
            trans += np.exp(compute_log_xi_sum(emission_probs, transition_est, forward, backward))
            total_lik += logsumexp(forward[-1])

        ems, probs = np.concatenate(all_emissions), np.concatenate(state_probs)

        initial_est = np.sum([p[0] for p in state_probs], axis=0)
        initial_est /= np.sum(initial_est)
        
        transition_est = normalize1(trans)
        means_est = compute_gaussian_mean(ems, probs)
        stds_est = compute_gaussian_std(means_est, ems, probs)
        print(total_lik)

    from hmmlearn import hmm
    emission_dump = np.concatenate(all_emissions)
    emission_lengths = [len(e) for e in all_emissions]
    
    model = hmm.GaussianHMM(n_components=Nstates, init_params='', n_iter=niter, params='mcst', covars_prior=0, covars_weight=0)
    model.startprob_ = initial_guess
    model.transmat_ = transition_guess
    model.covars_ = (stds_guess**2).reshape(-1, 1)
    model.means_ = means_guess.reshape(-1, 1)
    model.monitor_.tol = -np.inf
    model.monitor_.verbose = True
    model.fit(emission_dump.reshape(-1, 1), lengths=emission_lengths)
    
    """
    print(np.sqrt(model.covars_).ravel())
    print(stds_est)
    print(stds)
    """

    print("Cur", initial_est)
    print("Ref", model.startprob_)
    print("Tru", initial)

    state_prob2 = model.predict_proba(all_emissions[0].reshape(-1, 1))
    plt.plot(state_probs[0][:50], 'r')
    plt.plot(state_prob2[:50], 'g--')
    plt.show()

def verify_reest():
    np.random.seed(3)

    means = np.array([-1.0, 1.0, 2.0])
    stds = np.repeat(1.0, len(means))

    Nstates = len(means)
    
    trans = np.random.rand(Nstates, Nstates)
    trans += np.eye(*trans.shape)
    initial = np.random.rand(Nstates)
    initial /= np.sum(initial)
    trans /= np.sum(trans, axis=1).reshape(-1, 1)
    
    emission_dists = [scipy.stats.norm(m, s) for m, s in zip(means, stds)]
    
    states, emissions = simulate_process(100, initial, trans, emission_dists)
    
    #emission_probs = np.array([d.pdf(emissions) for d in emission_dists]).T
    #initial_guess = initial.copy()
    initial_guess = np.ones(Nstates)
    initial_guess /= np.sum(initial_guess)

    transition_guess = np.ones((Nstates, Nstates))
    #transition_guess = trans.copy()
    transition_guess /= np.sum(transition_guess, axis=1).reshape(-1, 1)
    
    means_guess = np.array([-1.0, 1.0, 2.0]) #+ np.random.randn(3)
    stds_guess = np.repeat(1.0, len(means))*0.5


    from hmmlearn import hmm
    transition_est = transition_guess
    initial_est = initial_guess.copy()
    means_est = means_guess.copy()
    stds_est = stds_guess.copy()
    niter = 3
    for itr in range(niter):
        emission_ests = [scipy.stats.norm(m, s) for m, s in zip(means_est, stds_est)]
        emission_probs = np.array([d.pdf(emissions) for d in emission_ests]).T
        #emission_probs /= np.sum(emission_probs, axis=1).reshape(-1, 1)
        state_prob, forward, backward = forward_backward(transition_est, emission_probs, initial_est)
        transition_est, initial_est = compute_transition_matrix(emission_probs, transition_est, state_prob, forward, backward)
        means_est, stds_est = compute_gaussian_emission_distribution(emissions, state_prob)

        #state_weights = state_prob/np.sum(state_prob, axis=0)
        #means_est = emissions@state_weights
        #diffs = emissions.reshape(-1, 1) - means_est
        #stds_est = np.sqrt(np.sum((diffs**2)*state_weights, axis=0))
        print(logsumexp(forward[-1]))
    emission_ests = [scipy.stats.norm(m, s) for m, s in zip(means_est, stds_est)]
    emission_probs = np.array([d.pdf(emissions) for d in emission_ests]).T
    state_prob, forward, backward = forward_backward(transition_est, emission_probs, initial_est)

    model = hmm.GaussianHMM(n_components=Nstates, init_params='', n_iter=niter, params='mcts', covars_prior=0, covars_weight=0)
    model.startprob_ = initial_guess
    model.transmat_ = transition_guess
    model.covars_ = (stds_guess**2).reshape(-1, 1)
    model.means_ = means_guess.reshape(-1, 1)
    model.monitor_.tol = -np.inf
    model.monitor_.verbose = True
    model.fit(emissions.reshape(-1, 1))
    print(model.monitor_)
    state_prob2 = model.predict_proba(emissions.reshape(-1, 1))

    print("Ref", np.sqrt(model.covars_).ravel())
    print("Cur", stds_est)
    
    print("Ref", (model.means_).ravel())
    print("Cur", means_est.ravel())

    plt.plot(state_prob, color='red')
    plt.plot(state_prob2, '--', color='green')
    plt.show()

def verify_forward_backward():
    np.random.seed(3)

    means = np.array([-1.0, 1.0, 2.0])*0.1
    stds = np.repeat(1.0, len(means))

    Nstates = len(means)
    
    trans = np.random.rand(Nstates, Nstates)
    trans += np.eye(*trans.shape)
    initial = np.random.rand(Nstates)
    initial /= np.sum(initial)
    trans /= np.sum(trans, axis=1).reshape(-1, 1)
    
    emission_dists = [scipy.stats.norm(m, s) for m, s in zip(means, stds)]

    states, emissions = simulate_process(10000, initial, trans, emission_dists)
    emission_probs = np.array([d.pdf(emissions) for d in emission_dists]).T
    
    state_prob, forward, backward = forward_backward(trans, emission_probs, initial)
    
    from hmmlearn import hmm

    model = hmm.GaussianHMM(n_components=Nstates)
    model.startprob_ = initial
    model.transmat_ = trans
    model.covars_ = (stds**2).reshape(-1, 1)
    model.means_ = means.reshape(-1, 1)

    print(np.sum(forward[-1]))
    state_prob2 = model.predict_proba(emissions.reshape(-1, 1))


    plt.plot(state_prob, color='red')
    plt.plot(state_prob2, '--', color='green')
    plt.show()

def demo():
    Nstates = 2
    trans = np.random.rand(Nstates, Nstates)
    trans += np.eye(*trans.shape)
    initial = np.random.rand(Nstates)
    initial /= np.sum(initial)
    trans /= np.sum(trans, axis=1).reshape(-1, 1)
    
    emissions = scipy.stats.norm(-1, 0.1), scipy.stats.norm(1, 0.1)

    states, emissions = simulate_process(100, initial, trans, emissions)

    plt.plot(emissions, '.-')
    plt.show()

if __name__ == '__main__':
    #demo()
    #verify_forward_backward()
    #verify_reest()
    verify_reest_multi()
    #demo_gohmm()
