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
    logging.debug("helper function")
    for i in range(len(raw_inputs)):
        for j in range(len(raw_inputs[i])):
            raw_inputs[i][j] = np.asarray(raw_inputs[i][j])
        raw_inputs[i] = np.asarray(raw_inputs[i])
        raw_outputs[i] = np.asarray(raw_outputs[i])
    for i in range(len(raw_outputs)):
        first_arr = np.abs(raw_inputs[i][1])
        second_arr = raw_outputs[i]

        # logging.debug(f"{first_arr.shape=}")
        # logging.debug(f"{second_arr.shape=}")

        heads_total = np.dot(first_arr, second_arr)
        # logging.debug(f"{heads_total=}")

        # logging.debug(f"{all_outputs=}")
        if (len(all_outputs) == 0):
            all_outputs = np.asarray([np.asarray([heads_total, sum(np.abs(raw_inputs[i][1])) - heads_total])])
        else:
            all_outputs = np.concatenate((all_outputs, [np.asarray([heads_total, sum(np.abs(raw_inputs[i][1])) - heads_total])]))

        # logging.debug(f"{all_outputs=}")
        # logging.debug(f"{heads_total=}")

        vec_inputs = np.abs(raw_inputs[i])
        # logging.debug(f"{vec_inputs=}")

        # logging.debug(f"{raw_outputs[i]=}")
        vec_inputs = np.append(vec_inputs, [raw_outputs[i]])
        # logging.debug(f"{vec_inputs=}")

        vec_inputs = np.hstack(vec_inputs)
        # logging.debug(f"{vec_inputs=}")

        if (len(all_inputs) == 0):
            all_inputs = np.asarray([vec_inputs])
        else:
            all_inputs = np.concatenate((all_inputs, [vec_inputs]))
        # logging.debug(f"{all_inputs=}")

def agg_convert_to_features(raw_inputs, raw_outputs):    
    global all_inputs, all_outputs, curr_round
    logging.debug("converting to features")
    if (curr_round == 0):
        agg_helper_featurize_outputs(raw_inputs, raw_outputs)
        logging.debug("finished agg_helper")
        all_outputs = all_outputs[1:]
    else:
        agg_helper_featurize_outputs(raw_inputs, raw_outputs)
    # return all_inputs[:-1], all_outputs
    return all_inputs, all_outputs

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

            if(len(raw_inputs) > 0):
                logger.info(f"IMPORTANT: {raw_inputs=}")
                logger.info(f"IMPORTANT: {raw_outputs=}")
                curr_inputs, curr_outputs = convert_to_features(featureType, raw_inputs, raw_outputs)
                logger.info(f"IMPORTANT: {curr_inputs=}")
                logger.info(f"IMPORTANT: {curr_outputs=}")

            if (len(curr_inputs[:-1]) > 0):
                # logging.debug(f"{curr_inputs=}")
                # logging.debug(f"{curr_outputs=}")
                # if (len(curr_outputs) > 5):
                #     curr_inputs = curr_inputs[-5:]
                #     curr_outputs = curr_outputs[-5:]
                clf = tree.DecisionTreeRegressor(max_depth=MAX_DEPTH)
                # logger.info("fitting model...")
                if (len(curr_inputs) > 100):
                    clf.fit(curr_inputs[-101:-1], curr_outputs[-100:])
                    sharedInfo.copy_decision_tree(clf)
                    logger.info("copying decision tree...")
                    
                    # Send model
                    logging.debug("IMPORTANT: test prediction")
                    # logging.debug(f"VERY CRUCIAL: Predicting: {clf.predict([curr_inputs[-1]])} compared to {curr_outputs[-1]}")
                    for i in range(0, len(curr_outputs)):
                        logging.debug(f"VERY CRUCIAL: Predicting: {clf.predict([curr_inputs[i]])} compared to {curr_outputs[i]}")
                    curr_inputs = []
                    curr_outputs = []
            curr_round += len(raw_outputs)
        except Exception as e:
            logger.error(e)