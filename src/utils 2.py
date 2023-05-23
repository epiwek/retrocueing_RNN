# utility function to generate data

import numpy as np
from collections import defaultdict

def generate_data(labels=False):
    """Generates a 2-hot matrix of color and cue data to train the model 
   
    Args:
        n_batches: sets how many batches to generate
        labels: if `True`, returns a list of the stimuli and cue per trial


    Returns:
        X and Y matrices. X is shape (32, 18, 10), representing the tuple
        (batch, timestep, feature). Y is shape (32, 10). If `labels` is True,
        also returns a list of the stimuli and cue used per trial."""
    
    # colors
    r = np.array([1, 0, 0, 0])
    b = np.array([0, 1, 0, 0])
    g = np.array([0, 0, 1, 0])
    y = np.array([0, 0, 0, 1])

    colors = r, b, g, y

    # cues
    u = np.array([1, 0])
    d = np.array([0, 1])
    
    cues = u, d

    # delay matrix
    delay = np.zeros(10)

    # make all experimental data
    # exp design: stimuli (500ms), delay (500ms), cue (300ms), delay (500ms)
    # with 1 timestep = 100ms
    xs = []
    ys = []
    if labels:
        def label(x):
            """Labels data."""
            names = {'red': r, 'blue': b, 'green': g, 'yellow': y,
                      'up': u, 'down': d}
            return next((k for k, v in names.items() if np.array_equal(v, x)))
        ls = []

    
    for color1 in colors:
        for color2 in colors:
            for cue in cues:
                if (cue == u).all():
                    ys.append(np.array(color1, dtype=float))
                elif (cue == d).all():
                    ys.append(np.array(color2, dtype=float))
                
                stimulus = np.concatenate((color1, color2, np.zeros(2)))
                if labels:
                    ls.append([label(color1), label(color2), label(cue)])
                
                # giving the cue a different name since we are 
                # altering it to fit the lstm input shape
                cue_input = np.concatenate((np.zeros(8), cue))
                trial = [
                    stimulus, stimulus, stimulus, stimulus, stimulus,
                    delay, delay, delay, delay, delay,
                    cue_input, cue_input, cue_input,
                    delay, delay, delay, delay, delay
                ]
                xs.append(np.array(trial, dtype=float))
                
    
    xs, ys = np.array(xs), np.array(ys)
    if labels:
        return xs, ys, ls
    else:
        return xs, ys