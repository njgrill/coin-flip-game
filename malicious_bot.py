import argparse
import socket
import sys
import re
import logging
import random

class Player:
    def __init__(self):
        self.choices = list() # Heads or Tails choices
        self.trades = list() # Amount traded
        self.total = 1000000


def parse_string(client, message):
    spltMsg = message[1:-1].split(':')
    newSpltMsg = [elem.split('=') for elem in spltMsg]
    result = newSpltMsg[3][1]
    client.actual_results.append(result)
    logging.info(newSpltMsg)

    client.total_flips = client.total_flips + 1
    client.num_heads = (client.num_heads + 1) if (result == "HEADS") else client.num_heads
    client.mle = (client.num_heads + .25 / client.total_flips + .5)

    for i in range (5, len(newSpltMsg), 3):
        #logging.info(newSpltMsg[i])
        username = newSpltMsg[i][1]

        if username == client.username:
            logging.info(f"{client.bet_amount=} {client.cur_strat=} {client.last_total=} cur_total={int(newSpltMsg[i+2][1])}")
            money_in_round = int(newSpltMsg[i+2][1]) - client.last_round_total
            client.last_round_total = int(newSpltMsg[i+2][1])
            money_diff = int(newSpltMsg[i+2][1]) - client.last_total
            if money_diff > 0:
                if client.total_flips < 400:
                    client.bet_amount += 40
                else:
                    client.bet_amount += 10
            else:
                if client.total_flips < 400:
                    client.bet_amount -= 10
                else:
                    client.bet_amount -= 3
                if client.bet_amount < 1000:
                    client.bet_amount = 1000
            if money_diff < -10000:
                logging.info("BIG CHUNGUS")
            if client.total_flips - client.last_check < 10:# and money_diff > -10000:
                return
            if int(newSpltMsg[i+2][1]) < client.last_total:
                client.cur_strat = "HEADS" if client.cur_strat == "TAILS" else "TAILS"
                logging.info(f"Bleeding cash: {client.cur_strat=}")
                client.bet_amount = 1000
            else:
                if money_diff / (client.total_flips - client.last_check) > 100:
                    client.bet_amount += 1000
            client.last_total = int(newSpltMsg[i+2][1])
            client.last_check = client.total_flips
            return
        else:
            client.usernames.add(username)




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
    if client.total_flips > 500:
        client.bet_multiplier = 10
    bet = client.bet_amount * client.bet_multiplier
    # if client.mle > 0.55 and client.cur_strat == "TAILS" or client.mle < 0.45 and client.cur_strat == "HEADS":
    #     bet = int(client.bet_amount / (1 + abs(.5-client.mle)))
    return client.cur_strat, max(int(bet), 1000)


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
    # logging.info("about to parse string")
    parse_string(client, message)
    logging.info(client.player_results)


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

        logging.basicConfig(filename=f"./logs/{game_id}/{parsed_args.output_logs}", encoding='utf-8', level=logging.DEBUG)
        logging.info(parsed_args)
        logging.info(f"{host} {port} {game_id} {username}")
        return Client(
            host=host,
            port=port,
            game_id=game_id,
            username=username,
            auction_request_hook=auction_request_hook,
            auction_result_hook=auction_result_hook
        )

    def __init__(self, host, port, game_id, username, auction_request_hook, auction_result_hook):
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
        logging.info("init")
        self.bet_multiplier = 1.5
        self.usernames = set()
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
        self.last_total = 1000000
        self.last_round_total = 1000000
        self.last_check = 0
        self.cur_strat = "HEADS"
        self.bet_amount = 1000
        self.mle = 0.5
        logging.info("done init")

    def run(self):
        """
           Log into the Auction Exchange Server and respond to auction requests.
        """
        logging.info("run")
        logging.info(f"{self.host} {self.port} {self.username} {self.game_id}")
        try:
            self.sock.connect((self.host, self.port))
        except Exception as e:
            logging.error(e)
        logging.info("connected")
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
        if not res:
            raise Exception("Server socket closed")
        received += res.decode("ascii")
        logging.info(received)
        match = self._MESSAGE_PATTERN.search(received)
        if match is None:
            return received

        message = match.group(0)
        message_type = match.group(1)
        logging.info(message)
        self._handle_server_message(message_type, message)
        received = received[len(message):]
        return received

    def _handle_server_message(self, message_type, message):
        if message_type == "AUCTION_REQUEST":
            auction_id = int(re.search('ID=(\d*)', message).group(1))
            ht = "HEADS"
            sz = 1000
            if len(self.usernames) == 0:
                ht, sz = self.auction_request_hook(self, auction_id)
            message = "|MT=AUCTION_RESPONSE:UN={}:GI={}:ID={}:HT={}:SZ={}|".format(
                self.username, self.game_id, auction_id, ht, sz
            )
            self._send_message(message)
            for username in self.usernames:
                message = "|MT=AUCTION_RESPONSE:UN={}:GI={}:ID={}:HT={}:SZ={}|".format(
                    username, self.game_id, auction_id, ht, 0
                )
                self._send_message(message)
        elif message_type == "AUCTION_RESULT":
            self.auction_result_hook(self, message)
        elif message_type == "REJECT":
            logging.error("Received reject message: " + message)
        if message_type == "END_OF_GAME":
            self.keep_running = False
            sys.exit(0)

    def _send_message(self, message):
        self.sock.sendall(bytes(message, "ascii"))

client = Client.initialize_client()
client.run()

