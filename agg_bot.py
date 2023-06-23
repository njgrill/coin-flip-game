import argparse
import copy
import math
import socket
import sys
import re
import random
import pickle
import time
import numpy as np
from threading import Thread
import logging
from copy import deepcopy


# from coin-flip-game.decision_tree import train_model;

from decision_tree import train_model

from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from shared_information import SharedInformation


class Player:
    def __init__(self):
        self.choices = list() # Heads or Tails choices
        self.trades = list() # Amount traded
        self.total = 1000000

def size_traded_to_amount_bet(client_amt_traded, size_traded, min_val: float = 1000.):
    # logging.debug(f"IMPORTANT: {size_traded=}")
    new_size_traded = abs(np.asarray(size_traded))
    # logging.debug(f"IMPORTANT: {new_size_traded=}")
    # pos_trades = np.asarray([elem for elem in size_traded if elem >= 0])
    pos_trades = [elem for elem in size_traded if elem >= 0]
    # logging.debug(f"IMPORTANT: {pos_trades=}")
    # logging.debug(f"IMPORTANT: {min_val=}")
    if (min(pos_trades) < min_val):
        if (min(pos_trades) == 0):
            pos_trades = [elem + min_val for elem in pos_trades]
        else:
            pos_trades_min = float(min(pos_trades))
            pos_trades = [elem * (min_val / pos_trades_min) for elem in pos_trades]
    # logging.debug(f"IMPORTANT: {size_traded=}")
    for i in range(len(size_traded)):
        if size_traded[i] >= 0:
            new_size_traded[i] = pos_trades.pop(0)
    logging.debug(f"IMPORTANT: {new_size_traded=}")
    return new_size_traded

def parse_string(client, message):
    spltMsg = message[1:-1].split(':')
    newSpltMsg = [elem.split('=') for elem in spltMsg]
    result = newSpltMsg[3][1]
    client.actual_results.append(result)
    client.logger.info(newSpltMsg)

    client.total_flips = client.total_flips + 1
    client.num_heads = (client.num_heads + 1) if (result == "HEADS") else (client.num_heads)

    client.mle = (client.num_heads + 1) / (client.total_flips + 2)

    cur_choice = []
    last_bet_size = []
    cur_total = []
    client.num_other_players = int((len(newSpltMsg) - 5)/3) - 1
    # other_result = "HEADS" if (result == "TAILS") else "TAILS"
    result_num = 1 if (result == "HEADS") else 0
    other_result_num = 0 if (result_num == 1) else 1
    client_choice_num = 1 if client.choice == "HEADS" else 0
    for i in range (5, len(newSpltMsg), 3):
        client.logger.info(newSpltMsg[i])
        username = newSpltMsg[i][1]
        client.logger.info(f"{username=}")
        size_traded = int(newSpltMsg[i+1][1])
        total = int(newSpltMsg[i+2][1])

        # if username not in client.player_results:
        #     client.player_results[username] = Player()
        # client.player_results[username].choices.append(result if size_traded > 0 else other_result)
        # logging.debug(client.player_results.get(username))
        # client.player_results[username].trades.append(size_traded)
        # client.player_results[username].total = int(newSpltMsg[i+2][1])

        if username != client.username:
            if (size_traded == 0):
                cur_choice.append((username, client_choice_num))
            else:
                cur_choice.append((username, result_num if size_traded > 0 else other_result_num))
            last_bet_size.append((username, size_traded))
            cur_total.append((username, total))
        else:
            client.size_traded = size_traded
            client.total = total

    logging.debug("finished range")
    cur_choice.sort()
    last_bet_size.sort()
    cur_total.sort()
    logging.debug(f"IMPORTANT: {client.choice=} and {cur_choice=}")
    logging.debug(f"IMPORTANT: {client.size_traded=} and {last_bet_size=}")
    cur_choice = [x[1] for x in cur_choice]
    min_val = client.last_bet if client.size_traded >= 0 else 1000
    last_bet_size = size_traded_to_amount_bet(client.size_traded, [x[1] for x in last_bet_size], min_val=min_val)
    cur_total = [x[1] for x in cur_total]

    client.shared_information.cur_round = client.total_flips
    # client.shared_information.add_to_queue([client.mle, last_bet_size, cur_total], cur_choice)
    client.shared_information.add_to_queue([client.mle, last_bet_size], cur_choice)

    logging.debug("about to make prediction")
    # vec_inputs = np.hstack([client.mle, last_bet_size, cur_total])
    vec_inputs = np.hstack([client.mle, last_bet_size])
    vec_inputs = np.append(vec_inputs, [cur_choice])

    vec_inputs = np.hstack(vec_inputs)
    logging.debug(f"about to maek {vec_inputs=}")
    # vec_inputs = vec_inputs.flatten()
    client.recent_predict = vec_inputs.reshape(1, -1)
    logging.debug(f"about to maek again {vec_inputs=}")
    client.logger.info(cur_choice)


def stub_handle_auction_request(client, auction_id: int) -> tuple[str, int]:
    """
       Default `auction_result_hook` argument of `Client.initialize_client`. Not currently implemented.

       Should be overwritten or replaced with a function that returns a tuple representing the client's bet for the auction round.

       Parameters
       ----------
       auction_id: int
           Auction round ID.

       Returns
       ----------
       str
           One of "HEADS" or "TAILS".
       int
           Wager size.
    """
    is_new, new_model = client.shared_information.get_decision_tree()
    logging.debug(f"{is_new=}, {new_model=}")
    if is_new:
        logging.debug("new model...")
        client.model = deepcopy(new_model)
        client.model_exists = True

    if client.model_exists:
        logging.debug(f"IMPORTANT: about to predict with {client.recent_predict=}")
        total_heads, total_tails = tuple(client.model.predict(client.recent_predict)[0])
        logging.debug(f"IMPORTANT: just predicted {total_heads=} and {total_tails=} and {client.num_other_players=} and {client.mle=}")
        avg_bet = (total_heads + total_tails) / client.num_other_players

        bet_amount = int(avg_bet)
        if bet_amount < 1000:
            bet_amount = 1000
        elif bet_amount > 50000:
            bet_amount = 50000
        if bet_amount > client.total:
            bet_amount = max(1000, int(client.total/2))

        ev_heads = (client.mle * (bet_amount / (total_heads + bet_amount)) * (total_tails)) + ((1. - client.mle) * ((-1. * bet_amount) if total_tails > 0 else 0))
        ev_tails = ((1. - client.mle) * (bet_amount / (total_tails + bet_amount)) * (total_heads)) + (client.mle * ((-1. * bet_amount) if total_heads > 0 else 0))
        client.last_bet = int(bet_amount)
        bet_choice = "HEADS" if ev_heads > ev_tails else "TAILS"
        # todo: have better amount stuff
        
        # DELETE THIS
        bet_amount = 5000

        if (ev_heads == ev_tails):
            client.choice = "HEADS" if client.mle >= 0.5 else "TAILS"
            logging.debug(f"IMPORTANT: Betting {client.choice=} with {ev_heads=} and {ev_tails=}")
            return ("HEADS", bet_amount) if client.mle >= 0.5 else ("TAILS", bet_amount)
        client.choice = bet_choice
        bet_amount = int(bet_amount)
        logging.debug(f"IMPORTANT: Betting {bet_choice} with {ev_heads=} and {ev_tails=}")
        return ("HEADS", bet_amount) if ev_heads > ev_tails else ("TAILS", bet_amount)
        # return "HEADS" if prediction == 1 else "TAILS", 1000
    else:
        client.last_bet = 5000
        client.choice = "HEADS" if client.mle >= 0.5 else "TAILS"
        logging.debug(f"IMPORTANT: Default bet...{client.choice=}")
        # return ("HEADS", 1000) if (client.mle >= 0.5) else ("TAILS", 1000)
        return ("HEADS", 5000) if (client.mle >= 0.5) else ("TAILS", 5000)


def stub_handle_auction_result(client, message: str) -> None:
    """
       Default `auction_result_hook` argument of `Client.initialize_client`. Prints `message`.

       Should be overwritten or replaced with a function that processes the `message` string.

       Parameters
       ----------
       message: str
           Raw `MT=AUCTION_RESULT` message sent by the Auction Exchange Server.
    """
    # ['MT', 'GI', 'ID', 'HT', 'TS', ['UN', 'SZ', 'TO'] ]
    parse_string(client, message)
    logging.debug(client.player_results)


class Client:
    _MESSAGE_PATTERN = re.compile('\|MT=(\w*):[\w=:-]*\|')

    @staticmethod
    def initialize_client(auction_request_hook=stub_handle_auction_request, auction_result_hook=stub_handle_auction_result, *args):
        """
           Helper method to initialize Client instance.

           Parameters
           ----------
           auction_request_hook: default: stub_handle_auction_request
               `auction_request_hook` method called by `Client`.
           auction_result_hook: default: stub_handle_auction_result
               `auction_result_hook` method called by `Client`.
           args: list of str
               String list args in the form of: `-un USERNAME host port game_id`. Can be set through code or command line.

           See Also
           ----------
           stub_handle_auction_request
           stub_handle_auction_result

           Returns
           ----------
           Client
        """

        parser = argparse.ArgumentParser()
        parser.add_argument("host")
        parser.add_argument("port")
        parser.add_argument("-gi", "--game_id", required=True)
        parser.add_argument("-un", "--username", required=True)
        parser.add_argument("-L", "--log_level", default=logging.DEBUG)
        parser.add_argument("-o", "--output_logs", required=True)

        parsed_args = parser.parse_args(*args)
        host = parsed_args.host
        port = int(parsed_args.port)
        game_id = parsed_args.game_id
        username = parsed_args.username

        logger = logging
        logger.basicConfig(filename=f"./logs/{game_id}/{parsed_args.output_logs}", encoding='utf-8', level=logging.DEBUG)
        logger.info(parsed_args)
        logger.info(f"{host} {port} {game_id} {username}")
        return Client(
            host=host,
            port=port,
            game_id=game_id,
            username=username,
            logger=logger,
            auction_request_hook=auction_request_hook,
            auction_result_hook=auction_result_hook
        )

    def __init__(self, host, port, game_id, username, logger, auction_request_hook, auction_result_hook):
        """
           Initialize an exchange client instance.

           Parameters
           ----------
           host: str
               The network location the Auction Exchange Server runs on.
           port: int
               The port exposed by the Auction Exchange Server for clients.
           game_id: int
               The ID of the auction.
           username: str
               The client's username.
           auction_request_hook
               Function hook that takes in the ID of the auction round and returns the clients bet as a tuple.
           auction_result_hook
               Function callback for the raw `MT=AUCTION_RESULT` message string received from the Auction Exchange Server.
        """
        self.host = host
        self.port = port
        self.game_id = game_id
        self.username = username
        self.keep_running = False
        self.auction_request_hook = auction_request_hook
        self.auction_result_hook = auction_result_hook
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.num_heads = 0
        self.total_flips = 0
        self.actual_results = list()
        self.player_results = dict()
        self.logger = logger
        self.shared_information = SharedInformation(self.logger)
        self.model_exists = False
        self.num_other_players = 1
        self.total = 1000000
        self.size_traded = 1000
        self.last_bet = 1000
        self.choice = "HEADS"
        self.mle = 0.5
        # modelThread = Thread(target = train_model, args= (self.shared_information, "Agg", logging,))
        modelThread = Thread(target = train_model, args = (self.shared_information, "Agg", self.logger,), daemon=True)
        modelThread.start()
        self.logger.info("done init")

    def run(self):
        """
           Log into the Auction Exchange Server and respond to auction requests.
        """
        logging.debug("run")
        logging.debug(f"{self.host} {self.port} {self.username} {self.game_id}")
        try:
            self.sock.connect((self.host, self.port))
        except Exception as e:
            logging.error(e)
        logging.debug("connected")
        logging.debug("|MT=LOGIN:GI={}:UN={}|".format(self.game_id, self.username))
        self._send_message("|MT=LOGIN:GI={}:UN={}|".format(self.game_id, self.username))

        received = ''
        self.keep_running = True
        while self.keep_running:
            try:
                received = self._run_loop(received)
            except Exception as ex:
                logging.error(ex)
                self.keep_running = False
        self.sock.shutdown(0)

    def _run_loop(self, received):
        res = self.sock.recv(1024)
        self.tic = time.perf_counter()
        if not res:
            raise Exception("Server socket closed")
        received += res.decode("ascii")
        logging.debug(received)
        match = self._MESSAGE_PATTERN.search(received)
        if match is None:
            return received

        message = match.group(0)
        message_type = match.group(1)
        logging.debug(message)
        self._handle_server_message(message_type, message)
        self.toc = time.perf_counter()
        logging.debug(f"Time to respond: {self.toc - self.tic:0.4f}")
        received = received[len(message):]
        return received

    def _handle_server_message(self, message_type, message):
        if message_type == "AUCTION_REQUEST":
            auction_id = int(re.search('ID=(\d*)', message).group(1))
            ht, sz = self.auction_request_hook(self, auction_id)
            message = "|MT=AUCTION_RESPONSE:UN={}:GI={}:ID={}:HT={}:SZ={}|".format(
                self.username, self.game_id, auction_id, ht, sz
            )
            logging.debug(f"IMPORTANT: Sending for round {auction_id=}")
            self._send_message(message)
        elif message_type == "AUCTION_RESULT":
            self.auction_result_hook(self, message)
        elif message_type == "REJECT":
            self.logger.error("Received reject message: " + message)
        if message_type == "END_OF_GAME":
            self.keep_running = False
            sys.exit(0)

    def _send_message(self, message):
        logging.debug(f"Socket message {message=}")
        self.sock.sendall(bytes(message, "ascii"))

client = Client.initialize_client()
client.run()

