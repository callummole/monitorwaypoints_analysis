import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import project_targets

colors = np.array([
        'r', 'g', 'b', 'k'
        ])

def safelog(x):
    return np.log(np.clip(x, 1e-9, None))

def viterbi(initial_probs, transition_probs, emissions):
    n_states = len(initial_probs)
    emissions = iter(emissions)
    emission = next(emissions)
    transition_probs = safelog(transition_probs)
    probs = safelog(emission) + safelog(initial_probs)
    state_stack = []
    
    for emission in emissions:
        emission /= np.sum(emission)
        trans_probs = transition_probs + np.row_stack(probs)
        most_likely_states = np.argmax(trans_probs, axis=0)
        probs = safelog(emission) + trans_probs[most_likely_states, np.arange(n_states)]
        state_stack.append(most_likely_states)
    
    state_seq = [np.argmax(probs)]

    while state_stack:
        most_likely_states = state_stack.pop()
        state_seq.append(most_likely_states[state_seq[-1]])

    state_seq.reverse()

    return state_seq

N_targets = 3
pitch = -np.radians(1.2225)*0
latency = 0.175
height = 1.2

def classify_gaze_targets(data):
    # Estimated using fit_gohmm
    cov = np.array([
        [1.95790476, 0.00313665],
        [0.00313665, 2.63977369]
        ])
    initial = np.array([0.74122825, 0.10438755, 0.09856582, 0.05581838])
    outlier_lik = 0.0001234567901234568
    trans = np.array([
        [9.83509885e-01, 9.95609801e-03, 4.49746443e-03, 2.03655304e-03],
        [1.11573550e-03, 9.80303046e-01, 9.57409556e-03, 9.00712253e-03],
        [1.20078465e-03, 4.20038960e-03, 9.80161859e-01, 1.44369670e-02],
        [4.33724131e-04, 1.83692097e-03, 6.27720007e-03, 9.91452155e-01]
        ])
    
    # Project targets
    for ti in range(N_targets):
        th = f"target_{ti}_h"
        tv = f"target_{ti}_v"
        
        data[th] = np.nan
        data[tv] = np.nan       

        geh = f"target_{ti}_geh"
        gev = f"target_{ti}_gev"


        
        # Project targets to visual angles
        for cond in data.condition.unique():
            targets = project_targets.COND_TARGETS[int(cond)]
            ci = data.condition == cond
            ni = (data.posx < 0) & ci
            data[th][ni], data[tv][ni] = map(np.degrees, project_targets.project_point(
                data.posx[ni], height, data.posz[ni],
                np.radians(data.yaw[ni]), pitch,
                *targets[ti]))
            
            ni = (data.posx > 0) & ci
            data[th][ni], data[tv][ni] = map(np.degrees, project_targets.project_point(
                data.posx[ni], height, data.posz[ni],
                np.radians(data.yaw[ni]), pitch,
                *targets[5-ti]))
        
        # Compute gaze error to the target
        data[geh] = data[th] - data.hangle
        data[gev] = data[tv] - data.vangle

        # A hack for the "outlier" target
        data[f"target_{N_targets}_h"] = np.nan
        data[f"target_{N_targets}_v"] = np.nan
    
    emission_distr = scipy.stats.multivariate_normal(cov=cov)
    data['target_class'] = N_targets
    for trial, td in data.groupby('trialcode'):
        emissions = [td[[f"target_{ti}_geh", f"target_{ti}_gev"]].values for ti in range(N_targets)]
        emissions = np.array(emissions)
        emissions = np.swapaxes(emissions, 0, 1)
        emission_probs = emission_distr.pdf(emissions)
        emission_probs = np.c_[emission_probs, np.full(len(emission_probs), outlier_lik)]
        states = viterbi(initial, trans, emission_probs)
        data.loc[td.index, "target_class"] = states

    return data
        

def classify_gaze_targets_OLD(data):
    for ti in range(N_targets):
        th = f"target_{ti}_h"
        tv = f"target_{ti}_v"
        
        data[th] = np.nan
        data[tv] = np.nan
        
        # Project targets to visual angles
        for cond in data.condition.unique():
            targets = project_targets.COND_TARGETS[int(cond)]
            ci = data.condition == cond
            ni = (data.posx < 0) & ci
            data[th][ni], data[tv][ni] = map(np.degrees, project_targets.project_point(
                data.posx[ni], height, data.posz[ni],
                np.radians(data.yaw[ni]), pitch,
                *targets[ti]))
            
            ni = (data.posx > 0) & ci
            data[th][ni], data[tv][ni] = map(np.degrees, project_targets.project_point(
                data.posx[ni], height, data.posz[ni],
                np.radians(data.yaw[ni]), pitch,
                *targets[5-ti]))
        
        # A hack for the "outlier" target
        data[f"target_{N_targets}_h"] = np.nan
        data[f"target_{N_targets}_v"] = np.nan
    
    # Set up the transitions
    N_states = N_targets + 1
    initial_probs = np.ones(N_states)
    initial_probs /= np.sum(initial_probs)
    transition_probs = np.ones((N_states, N_states))
    # Assume it's 10 times as likely to stay on the current target
    # as it is to switch targets. This could be a lot nicer
    np.fill_diagonal(transition_probs, 10)
    transition_probs /= np.sum(transition_probs, axis=0)

    # Set up the observation model. Currently just Stetson-Harrison
    hmodel = scipy.stats.norm(0.0, 5.0)
    vmodel = scipy.stats.norm(0.0, 10.0)

    # By default set the target class to "outlier"
    data['target_class'] = N_states
    
    # Compute state likelihoods for the outlier state. Could be
    # nicer.
    outlik = hmodel.pdf(5)*vmodel.pdf(10)

    for trial, td in data.groupby('trialcode'):
        # Compute the observation likelihoods
        liks = []
        for i in range(N_targets):
            th = td[f"target_{i}_h"]
            tv = td[f"target_{i}_v"]
            herrors = th - td.hangle
            verrors = tv - td.vangle
            lik = hmodel.pdf(herrors)*vmodel.pdf(verrors)
            liks.append(lik)
        
        liks.append(np.repeat(outlik, len(td)))
        liks = np.array(liks).T
        
        # Run the classification
        states = viterbi(initial_probs, transition_probs, liks)
        data.loc[td.index, "target_class"] = states
    return data

def forward_backward(transition_probs, observations, initial_probs=None):
    observations = np.array(list(observations))
    N = len(transition_probs)
    T = len(observations)
    if initial_probs is None:
        initial_probs = np.ones(N)
        initial_probs /= np.sum(initial_probs)
    
    forward_probs = np.zeros((T, N))
    backward_probs = forward_probs.copy()
    probs = initial_probs
    for i in range(T):
        probs = np.dot(probs, transition_probs)*observations[i]
        probs /= np.sum(probs)
        forward_probs[i] = probs
    
    probs = np.ones(N)
    probs /= np.sum(probs)
    for i in range(T-1, -1, -1):
        probs = np.dot(transition_probs, (probs*observations[i]).T)
        probs /= np.sum(probs)
        backward_probs[i] = probs
    
    state_probs = forward_probs*backward_probs
    state_probs /= np.sum(state_probs, axis=1).reshape(-1, 1)
    return state_probs, forward_probs, backward_probs

def transition_estimates(obs, trans, forward, backward):
    T, N = len(obs), len(trans)
    ests = np.zeros((T, N, N))
    for start, end, i in itertools.product(range(N), range(N), range(T)):
        if i == T - 1:
            b = 1/N
        else:
            b = backward[i+1, end]
        ests[i,start,end] = forward[i,start]*b*trans[start,end]
    return ests

def fit_gohmm(data):
    # Project targets
    for ti in range(N_targets):
        th = f"target_{ti}_h"
        tv = f"target_{ti}_v"
        
        data[th] = np.nan
        data[tv] = np.nan       

        geh = f"target_{ti}_geh"
        gev = f"target_{ti}_gev"


        
        # Project targets to visual angles
        for cond in data.condition.unique():
            targets = project_targets.COND_TARGETS[int(cond)]
            ci = data.condition == cond
            ni = (data.posx < 0) & ci
            data[th][ni], data[tv][ni] = map(np.degrees, project_targets.project_point(
                data.posx[ni], height, data.posz[ni],
                np.radians(data.yaw[ni]), pitch,
                *targets[ti]))
            
            ni = (data.posx > 0) & ci
            data[th][ni], data[tv][ni] = map(np.degrees, project_targets.project_point(
                data.posx[ni], height, data.posz[ni],
                np.radians(data.yaw[ni]), pitch,
                *targets[5-ti]))
        
        # Compute gaze error to the target
        data[geh] = data[th] - data.hangle
        data[gev] = data[tv] - data.vangle
    
    trials = []
    for trial, td in data.groupby('trialcode'):
        emissions = [td[[f"target_{ti}_geh", f"target_{ti}_gev"]].values for ti in range(N_targets)]
        emissions = np.array(emissions)
        emissions = np.swapaxes(emissions, 0, 1)
        trials.append(emissions)

    from gohmm import normalize1, forward_backward, logsumexp, compute_log_xi_sum

    
    Nstates = N_targets + 1
    # Start with equal proability of being in each state
    initial = np.ones(Nstates)/Nstates
    # And equal probability of transitioning between states. We can probably
    # have a better guess, but this is at least very uninformative
    trans = normalize1(np.ones((Nstates, Nstates)))

    # Start with guess of gaze error standard deviation being 5 degrees
    # from targets with independent axes
    cov = np.sqrt(np.eye(2)*5)

    # Assume 90 x 90 degree fov for the outliers. This is not really computed from
    # a distribution and values can go over it, but it gives a sort of intuitive meaning
    # for the outlier state
    outlier_lik = (1/90)**2

    niters = 100
    for itr in range(niters):
        emission_distr = scipy.stats.multivariate_normal(cov=cov)
        
        total_lik = 0
        initials = np.zeros(Nstates)
        cumtrans = np.zeros((Nstates, Nstates))
        # TODO: Storing all observations now to compute the covariance matrix
        # later. This can be done online, but I haven't figured out how
        allobs = []
        allobsps = []

        for trial in trials:
            # Compute emission probabilities for each state
            emission_probs = emission_distr.pdf(trial)
            # And add the outlier state
            emission_probs = np.c_[emission_probs, np.full(len(emission_probs), outlier_lik)]
            
            # Compute the HMM probability stuff
            state_probs, forward, backward = forward_backward(trans, emission_probs, initial)
            
            # Add out likelihood to the total
            total_lik += logsumexp(forward[-1])

            # Add contribution to initial state probability
            initials += state_probs[0]
            # and the transitions
            logxi_sum = compute_log_xi_sum(emission_probs, trans, forward, backward)
            cumtrans += np.exp(logxi_sum)
            
            # Store the observations and their probabilities for each target
            # to compute the covariance matrix. This can be done a lot faster
            for target in range(N_targets):
                ps = state_probs[:,target]
                obs = trial[:,target]
                allobs.extend(obs)
                allobsps.extend(ps)

 

        # Update the initial state estimate
        initial = initials/np.sum(initials)
        # and the state transition estimate
        trans = normalize1(cumtrans)

        # and the covariance matrix.
        # TOOD: This is really slow way of doing this
        allobsps = np.array(allobsps)
        allobsps /= np.sum(allobsps)
        allobs = np.array(allobs)
        wobs = allobsps.reshape(-1, 1)*allobs
        cov = wobs.T@allobs
        unbias = 1/(1 - np.sum(allobsps**2))
        cov *= unbias
    
    return initial, trans, cov, outlier_lik

def demo_gohmm():
    data = pd.read_feather("../Data/trout_6.feather")
    data = data.query("dataset == 2 and roadsection == '2' and obstaclecolour == '0'")
    initial, trans, cov, outlier_lik = fit_gohmm(data)
    from pprint import pprint
    pprint(dict(
        initial=initial,
        trans=trans,
        cov=cov,
        outlier_lik=outlier_lik))

def demo():
    data = pd.read_feather("../Data/trout_6.feather")
    
    # Extract only attract-trials
    data = data.query("dataset == 2 and roadsection == '2' and obstaclecolour == '0'").copy()
    
    for sid, sd in data.groupby('ID'):
        sd = classify_gaze_targets(sd)
        
        for trial, td in sd.groupby('trialcode'):
            fig, (vax, hax) = plt.subplots(ncols=2)

            # Do some plotting
            for i in range(N_targets):
                th = td[f"target_{i}_h"]
                tv = td[f"target_{i}_v"]
            
                valid = (np.abs(td[f"target_{i}_h"]) < 30) & (np.abs(td[f"target_{i}_v"]) < 30)
                vax.plot(td.currtime[valid], td[f"target_{i}_v"][valid], alpha=0.3, color=colors[i], label=str(i))
                hax.plot(td.currtime[valid], td[f"target_{i}_h"][valid], alpha=0.3, color=colors[i], label=str(i))
            
            states = td['target_class'].values
            for state in np.unique(states):
                my = states == state
                vax.plot(td.currtime.values[my], td.vangle.values[my], '.', color=colors[state])
                hax.plot(td.currtime.values[my], td.hangle.values[my], '.', color=colors[state])

            plt.suptitle((trial, td.condition.iloc[0], td.obstaclecolour.iloc[0], td.drivingmode.iloc[0]))
            plt.show()

def oldmain():
    data = pd.read_parquet("steergazedata_fulltrials.parquet")
    
    # Extract only attract-trials
    data = data.query("roadsection == '2' and obstaclecolour == '0'").copy()
    
    for ti in range(N_targets):
        th = f"target_{ti}_h"
        tv = f"target_{ti}_v"
        
        data[th] = np.nan
        data[tv] = np.nan
        
        for cond in data.condition.unique():
            targets = project_targets.COND_TARGETS[int(cond)]
            ci = data.condition == cond
            ni = (data.posx < 0) & ci
            data[th][ni], data[tv][ni] = map(np.degrees, project_targets.project_point(
                data.posx[ni], 1.2, data.posz[ni],
                np.radians(data.yaw[ni]), pitch,
                *targets[ti]))
            
            ni = (data.posx > 0) & ci
            data[th][ni], data[tv][ni] = map(np.degrees, project_targets.project_point(
                data.posx[ni], 1.2, data.posz[ni],
                np.radians(data.yaw[ni]), pitch,
                *targets[5-ti]))
    
    # A hack
    data[f"target_{N_targets}_h"] = np.nan
    data[f"target_{N_targets}_v"] = np.nan
    
    N_states = N_targets + 1
    initial_probs = np.ones(N_states)
    initial_probs /= np.sum(initial_probs)
    transition_probs = np.ones((N_states, N_states))
    np.fill_diagonal(transition_probs, 10)
    transition_probs /= np.sum(transition_probs, axis=0)
    
    hmodel = scipy.stats.norm(0.0, 5.0)
    vmodel = scipy.stats.norm(0.0, 10.0)
    outlik = hmodel.pdf(5)*vmodel.pdf(10)
    
    data['hangle_orig'] = data['hangle']
    data['vangle_orig'] = data['vangle']
    data['target_class'] = 3
    for giter in range(10):
        error_stds = []
        for sid, sd in data.groupby('ID'):
            for trial, td in sd.groupby('trialcode'):
                fig, (vax, hax) = plt.subplots(ncols=2)
                liks = []
                for i in range(N_targets):
                    #th, tv = project_targets.project_point(
                    #    td.posx, 1.2, td.posz,
                    #    np.radians(td.yaw),
                    #    *target)
                    #th, tv = target_angles[i]
                    #th, tv = map(np.degrees, (th, tv))
                    th = td[f"target_{i}_h"]
                    tv = td[f"target_{i}_v"]
                    herrors = th - td.hangle
                    verrors = tv - td.vangle
                    lik = hmodel.pdf(herrors)*vmodel.pdf(verrors)
                    liks.append(lik)
                
                    valid = (np.abs(td[f"target_{i}_h"]) < 30) & (np.abs(td[f"target_{i}_v"]) < 30)
                    vax.plot(td.currtime[valid], td[f"target_{i}_v"][valid], alpha=0.3, color=colors[i], label=str(i))
                    hax.plot(td.currtime[valid], td[f"target_{i}_h"][valid], alpha=0.3, color=colors[i], label=str(i))
                
                liks.append(np.repeat(outlik, len(td)))
                liks = np.array(liks).T
                states = viterbi(initial_probs, transition_probs, liks)
                
                preds = np.array([td[[f"target_{s}_h",f"target_{s}_v"]].iloc[i].values for i, s in enumerate(states)])
                herror = td.hangle - preds[:,0]
                verror = td.vangle - preds[:,1]

                error_stds.append((np.nanstd(herror), np.nanstd(verror)))
                for state in np.unique(states):
                    my = states == state
                    vax.plot(td.currtime.values[my], td.vangle.values[my], '.', color=colors[state])
                    hax.plot(td.currtime.values[my], td.hangle.values[my], '.', color=colors[state])

                plt.title((trial, td.condition.iloc[0], td.obstaclecolour.iloc[0], td.drivingmode.iloc[0]))
                hax.set_ylim(-25, 25)
                vax.set_ylim(-25, 25)
                plt.show()
                
                data.loc[td.index, "target_class"] = states
                data.loc[td.index, "hangle"] -= np.nanmean(herror)
                data.loc[td.index, "vangle"] -= np.nanmean(verror)
                

        error_stds = np.array(error_stds, axis=0)
        print(error_stds)


if __name__ == '__main__':
    demo()
    #demo_gohmm()
