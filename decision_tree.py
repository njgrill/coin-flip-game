import numpy as np
import logging
from shared_information import SharedInformation
from sklearn import tree

MAX_DEPTH = None
all_inputs = np.asarray([])
all_outputs = np.asarray([])
curr_round = 0

def agg_helper_featurize_outputs(raw_inputs, raw_outputs):
    global all_inputs, all_outputs, curr_round
    logging.info("helper function")
    for i in range(len(raw_inputs)):
        for j in range(len(raw_inputs[i])):
            raw_inputs[i][j] = np.asarray(raw_inputs[i][j])
        raw_inputs[i] = np.asarray(raw_inputs[i])
        raw_outputs[i] = np.asarray(raw_outputs[i])
    for i in range(len(raw_outputs)):
        first_arr = np.abs(raw_inputs[i][1])
        second_arr = raw_outputs[i]

        logging.info(f"{first_arr.shape=}")
        logging.info(f"{second_arr.shape=}")

        heads_total = np.dot(first_arr, second_arr)
        logging.info(f"{heads_total=}")

        logging.info(f"{all_outputs=}")
        if (len(all_outputs) == 0):
            all_outputs = np.asarray([np.asarray([heads_total, sum(np.abs(raw_inputs[i][1])) - heads_total])])
        else:
            all_outputs = np.concatenate((all_outputs, [np.asarray([heads_total, sum(np.abs(raw_inputs[i][1])) - heads_total])]))

        logging.info(f"{all_outputs=}")
        logging.info(f"{heads_total=}")

        vec_inputs = np.abs(raw_inputs[i])
        logging.info(f"{vec_inputs=}")

        logging.info(f"{raw_outputs[i]=}")
        vec_inputs = np.append(vec_inputs, [raw_outputs[i]])
        logging.info(f"{vec_inputs=}")

        vec_inputs = np.hstack(vec_inputs)
        logging.info(f"{vec_inputs=}")

        if (len(all_inputs) == 0):
            all_inputs = np.asarray([vec_inputs])
        else:
            all_inputs = np.concatenate((all_inputs, [vec_inputs]))
        logging.info(f"{all_inputs=}")

def agg_convert_to_features(raw_inputs, raw_outputs):    
    global all_inputs, all_outputs, curr_round
    logging.info("converting to features")
    if (curr_round == 0):
        agg_helper_featurize_outputs(raw_inputs, raw_outputs)
        logging.info("finished agg_helper")
        all_outputs = all_outputs[1:]
    else:
        agg_helper_featurize_outputs(raw_inputs, raw_outputs)
    return all_inputs[:-1], all_outputs

def convert_to_features(featureType, raw_inputs, raw_outputs):
    if featureType == "Agg":
        return agg_convert_to_features(raw_inputs, raw_outputs)

# Inputs: n choices, n amounts, n totals
# Outputs: [total HEADS bets, total TAILS bets]
def train_model(sharedInfo: SharedInformation, featureType: str, logger):
    global all_inputs, all_outputs, curr_round
    curr_inputs = []
    curr_outputs = []
    while(True):
        try:
            # logger.info("starting train_model")
            raw_inputs, raw_outputs = sharedInfo.copy_queue()
            # logger.info(f"{raw_inputs=}")
            # logger.info(f"{raw_outputs=}")

            if(len(raw_inputs) > 0):
                curr_inputs, curr_outputs = convert_to_features(featureType, raw_inputs, raw_outputs)

            if (len(curr_inputs) > 0):
                logging.info(f"{curr_inputs=}")
                logging.info(f"{curr_outputs=}")
                clf = tree.DecisionTreeRegressor(max_depth=MAX_DEPTH)
                logger.info("fitting model...")
                clf.fit(curr_inputs, curr_outputs)
                
                # Send model
                logging.info("test prediction")
                logging.info(clf.predict([curr_inputs[0]]))
                logger.info("copying decision tree...")
                sharedInfo.copy_decision_tree(clf)
                curr_inputs = []
                curr_outputs = []
            curr_round += len(raw_outputs)
        except Exception as e:
            logger.error(e)