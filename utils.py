import numpy as np
from copy import deepcopy

try:
    # Python 2 module
    from itertools import izip_longest as zip_longest
except ImportError:
    # Python 3 module
    from itertools import zip_longest

def add_conv(layers, max_out_ch, conv_kernel):
    out_channel = np.random.randint(3, max_out_ch)
    conv_kernel = np.random.randint(3, conv_kernel)

    layers.append({"type": "conv", "ou_c": out_channel, "kernel": conv_kernel})

    return layers

def add_res(layers, max_out_ch, conv_kernel):
    out_channel = np.random.randint(3, max_out_ch)
    conv_kernel = np.random.randint(3, conv_kernel)

    layers.append({"type": "res", "ou_c": out_channel, "kernel": conv_kernel})

    return layers


def add_fc(layers, max_fc_neurons):
    layers.append({"type": "fc", "ou_c": np.random.randint(1, max_fc_neurons), "kernel": -1})
    
    return layers
        

def add_pool(layers, fc_prob, num_pool_layers, max_pool_layers, max_out_ch, max_conv_kernel, max_fc_neurons, output_dim):
    pool_layers = num_pool_layers

    if pool_layers < max_pool_layers:
        random_pool = np.random.rand()
        pool_layers += 1
        if random_pool < 0.5:
            # Add Max Pooling
            layers.append({"type": "max_pool", "ou_c": -1, "kernel": 2})
        else:
            layers.append({"type": "avg_pool", "ou_c": -1, "kernel": 2})
    
    return layers, pool_layers


def differenceConvPool(p1, p2):
    diff = []

    for comb in zip_longest(p1, p2):
        if comb[0] != None and comb[1] != None:
            if comb[0]["type"] == comb[1]["type"]:
                diff.append({"type": "keep"})
            else:
                diff.append(comb[0])

        elif comb[0] != None and comb[1] == None:
            diff.append(comb[0])

        elif comb[0] == None and comb[1] != None:
            diff.append({"type": "remove"})
    
    return diff


def differenceFC(p1, p2):
    diff = []

    # Compute the difference from the end to the begin
    for comb in zip_longest(p1[::-1], p2[::-1]):
        if comb[0] != None and comb[1] != None:
            diff.append({"type": "keep_fc"})
        elif comb[0] != None and comb[1] == None:
            diff.append(comb[0])
        elif comb[0] == None and comb[1] != None:
            diff.append({"type": "remove_fc"})

    diff = diff[::-1]
    
    return diff


def computeDifference(p1, p2):
    diff = []
    # First, find the index where the fully connected layers start in each particle
    p1fc_idx = next((index for (index, d) in enumerate(p1) if d["type"] == "fc"))
    p2fc_idx = next((index for (index, d) in enumerate(p2) if d["type"] == "fc"))

    # Compute the difference only between the convolution and pooling layers
    diff.extend(differenceConvPool(p1[0:p1fc_idx], p2[0:p2fc_idx]))
    
    # Compute the difference between the fully connected layers 
    diff.extend(differenceFC(p1[p1fc_idx:], p2[p2fc_idx:]))
    
    keep_all_layers = True
    for i in range(len(diff)):
        if diff[i]["type"] != "keep" or diff[i]["type"] != "keep_fc":
            keep_all_layers = False
            break

    return diff, keep_all_layers

def velocityConvPool(diff_pBest, diff_gBest, Cg):
    vel = []

    for comb in zip_longest(diff_pBest, diff_gBest):
        if np.random.uniform() <= Cg:
            if comb[1] != None:
                vel.append(comb[1])
            else:
                vel.append({"type": "remove"})
        else:
            if comb[0] != None:
                vel.append(comb[0])
            else:
                vel.append({"type": "remove"})

    return vel

def velocityFC(diff_pBest, diff_gBest, Cg):
    vel = []

    for comb in zip_longest(diff_pBest[::-1], diff_gBest[::-1]):
        if np.random.uniform() <= Cg:
            if comb[1] != None:
                vel.append(comb[1])
            else:
                vel.append({"type": "remove_fc"})
        else:
            if comb[0] != None:
                vel.append(comb[0])
            else:
                vel.append({"type": "remove_fc"})
    
    vel = vel[::-1]

    return vel


def computeVelocity(gBest, pBest, p, Cg):
    diff_pBest, keep_all_pBest = computeDifference(pBest, p)
    diff_gBest, keep_all_gBest = computeDifference(gBest, p)

    velocity = []

    # First, verify if the general architecture is the same in both difference
    if keep_all_pBest == True and keep_all_gBest == True:
        for i in range(len(gBest)):
            if np.random.uniform () <= Cg:
                velocity.append(gBest[i])
            else:
                velocity.append(pBest[i])
    else:
        # Find the index where the fully connected layers start in each particle
        dp_fc_idx = next((index for (index, d) in enumerate(diff_pBest) if d["type"] == "fc" or d["type"] == "keep_fc" or d["type"] == "remove_fc"))
        dg_fc_idx = next((index for (index, d) in enumerate(diff_gBest) if d["type"] == "fc" or d["type"] == "keep_fc" or d["type"] == "remove_fc"))

        # Compute the velocity only between the convolution and pooling layers
        velocity.extend(velocityConvPool(diff_pBest[0:dp_fc_idx], diff_gBest[0:dg_fc_idx], Cg))

        # Compute the velocity between the fully connected layers
        velocity.extend(velocityFC(diff_pBest[dp_fc_idx:], diff_gBest[dg_fc_idx:], Cg))

    return velocity


def updateConvPool(p, vel):
    new_p = []

    for comb in zip_longest(p, vel):
        if comb[1]["type"] != "remove":
            if comb[1]["type"] == "keep":
                new_p.append(comb[0])
            else:
                new_p.append(comb[1])

    return new_p


def updateFC(p, vel):
    new_p = []

    for comb in zip_longest(p[::-1], vel[::-1]):
        if comb[1]["type"] != "remove_fc":
            if comb[1]["type"] == "keep_fc":
                new_p.append(comb[0])
            else:
                new_p.append(comb[1])

    new_p = new_p[::-1]

    return new_p


def updateParticle(p, velocity):
    new_p = []

    dp_fc_idx = next((index for (index, d) in enumerate(p) if d["type"] == "fc"))
    dg_fc_idx = next((index for (index, d) in enumerate(velocity) if d["type"] == "fc" or d["type"] == "keep_fc" or d["type"] == "remove_fc"))

    # Update only convolution and pooling layers
    new_p.extend(updateConvPool(p[0:dp_fc_idx], velocity[0:dg_fc_idx]))

    # Update only fully connected layers
    new_p.extend(updateFC(p[dp_fc_idx:], velocity[dg_fc_idx:]))
        
    return new_p
