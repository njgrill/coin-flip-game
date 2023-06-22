import numpy as np
import logging
from shared_information import SharedInformation
from sklearn import tree

MAX_DEPTH = None
all_inputs = []
all_outputs = []
curr_round = 0

def agg_helper_featurize_outputs(raw_inputs, raw_outputs):
    for i in range(len(raw_outputs)):
        heads_total = np.dot(np.abs(raw_inputs[i][1]), raw_outputs[i])
        all_outputs.append([heads_total, sum(np.abs(raw_inputs[i][1])) - heads_total])
        vec_inputs = raw_inputs[i].append(raw_outputs[i]).flatten()
        all_inputs.append(vec_inputs)

def agg_convert_to_features(raw_inputs, raw_outputs):    
    if (curr_round == 0):
        agg_helper_featurize_outputs(raw_inputs, raw_outputs)
        all_outputs.pop(0)
    else:
        agg_helper_featurize_outputs(raw_inputs, raw_outputs)
    return all_inputs[:-1], all_outputs

def convert_to_features(featureType, raw_inputs, raw_outputs):
    if featureType == "Agg":
        agg_convert_to_features(raw_inputs, raw_outputs)

# Inputs: n choices, n amounts, n totals
# Outputs: [total HEADS bets, total TAILS bets]
def train_model(sharedInfo: SharedInformation, featureType: str, logger):
    while(True):
        logger.info("starting train_model")
        raw_inputs, raw_outputs = sharedInfo.copy_queue()

        if(len(raw_inputs) > 0):
            curr_inputs, curr_outputs = convert_to_features(featureType, raw_inputs, raw_outputs)

        if (len(curr_inputs) > 0):
            clf = tree.DecisionTreeRegressor(max_depth=MAX_DEPTH)
            clf = clf.fit(all_inputs, all_outputs)

            # Send model
            sharedInfo.copy_decision_tree(clf)
            curr_round += len(raw_outputs)