import pandas as pd
import matplotlib.pyplot as plt
import project_targets
import numpy as np
import scipy.stats
import sys

#data = pd.read_parquet("steergazedata_fulltrials.parquet")
data = pd.read_csv(sys.argv[1])

condition = '1'
data.query(
        "roadsection == '2' and obstaclecolour == '0' and condition==@condition",
        inplace=True)

print(data.columns)
data['hangle'] = data['hangle_new']
data['vangle'] = data['vangle_new']
alltargets = project_targets.COND_TARGETS[int(condition)]
#alltargets = targets[-3:][::-1]

def safelog(x):
    return np.log(np.clip(x, 1e-6, None))

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

pitch = np.radians(1.2225)
latency = 0
#hmodel = scipy.stats.norm(0.0, 5.0)
#vmodel = scipy.stats.norm(0.0, 10.0)
dist_family = scipy.stats.norm
N = 3 + 1
initial_probs = np.ones(N)
#initial_probs[[0, -1]] = 10
initial_probs /= np.sum(initial_probs)
transition_probs = np.ones((N, N))
np.fill_diagonal(transition_probs, 10)
transition_probs /= np.sum(transition_probs, axis=0)
print(transition_probs)

colors = np.array([
        'r', 'g', 'b', 'k'
        ])
from matplotlib.backends.backend_pdf import PdfPages

#pdf = PdfPages("gazetargets.pdf")
#plt.show = lambda: (pdf.savefig(), plt.close)

for (sid, block, t), td in data.groupby(['ID', 'block', 'trialcode']):
    if td.posx.iloc[0] > 0:
        targets = alltargets[-3:][::-1]
    else:
        targets = alltargets[:3]
    target_angles = []
    for i, target in enumerate(targets):
            th, tv = project_targets.project_point(
                    td.posx, 1.2, td.posz,
                    np.radians(td.yaw), pitch,
                    *target)
            
            target_angles.append((np.degrees(th), np.degrees(tv)))
    
    # A hack
    target_angles.append((
        np.repeat(np.nan, len(td)),
        np.repeat(np.nan, len(td))
        ))
    target_angles = np.array(target_angles)
    
    hmodel = dist_family(0.0, 5.0)
    vmodel = dist_family(0.0, 10.0)
    outlik = hmodel.pdf(5)*vmodel.pdf(10)


    otd = td.copy()
    nitr = 1
    for itr in range(nitr):

        liks = []
        _, (hax, vax) = plt.subplots(2, 1, sharex=True)
        
        hax.plot(otd.currtimezero, otd.hangle, alpha=0.3, color='black', label="Orig gaze")
        vax.plot(otd.currtimezero, otd.vangle, alpha=0.3, color='black', label="Orig gaze")
        for i, target in enumerate(targets):
            th, tv = target_angles[i]
            herrors = th - td.hangle
            verrors = tv - td.vangle
            lik = hmodel.pdf(herrors)*vmodel.pdf(verrors)
            liks.append(lik)

            valid = (np.abs(tv) < 30) & (np.abs(th) < 30)

            hax.plot(td.currtimezero[valid], th[valid], alpha=0.3, color=colors[i], label=f"Target {i}")
            vax.plot(td.currtimezero[valid], tv[valid], alpha=0.3, color=colors[i])
       
        liks.append(np.repeat(outlik, len(td)))
        liks = np.array(liks).T
        states = viterbi(initial_probs, transition_probs, liks)
        state_probs = forward_backward(transition_probs, liks, initial_probs)[0]

        
        #preds = np.take_along_axis(target_angles, states, axis=2)
        #print(preds.shape)
        hax.scatter(td.currtimezero, td.hangle, c=colors[states], marker='.', label='Gaze')
        vax.scatter(td.currtimezero, td.vangle, c=colors[states], marker='.', label='Gaze')
        
        vax.set_xlabel('Time (seconds)')
        vax.set_ylabel('Vert. gaze (deg)')
        hax.set_ylabel('Horiz. gaze (deg)')
        #vax.scatter(td.currtime, td.vangle, c=state_probs[:,:3], marker='.')
        #hax.scatter(td.currtime, td.hangle, c=state_probs[:,:3], marker='.')

        hax.set_title(t)
        
        hax.legend()
        vax.set_ylim(-30, 10)
        hax.set_ylim(-20, 20)
        if itr < nitr - 1:
            plt.close()
        else:
            plt.show()
        
        preds = np.array([target_angles[states[i],:,i] for i in range(len(target_angles[0,0]))])
        herror = td.hangle - preds[:,0]
        verror = td.vangle - preds[:,1]
        
        outlik = 1
        mederror = np.nanmean(herror)
        mad = np.nanstd(herror - mederror)
        td.hangle -= mederror
        hmodel = dist_family(0.0, mad)
        outlik *= hmodel.pdf(mad*3)

        mederror = np.nanmean(verror)
        mad = np.nanstd(verror - mederror)
        td.vangle -= mederror
        vmodel = dist_family(0.0, mad)
        outlik *= vmodel.pdf(mad*3)
pdf.close()
