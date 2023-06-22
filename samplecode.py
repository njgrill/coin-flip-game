import argparse
import socket
import sys
import re


def stub_handle_auction_request(auction_id: int) -> tuple[str, int]:
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
    raise NotImplemented("Put your implementation here")


def stub_handle_auction_result(message: str) -> None:
    """
       Default `auction_result_hook` argument of `Client.initialize_client`. Prints `message`.

       Should be overwritten or replaced with a function that processes the `message` string.

       Parameters
       ----------
       message: str
           Raw `MT=AUCTION_RESULT` message sent by the Auction Exchange Server.
    """
    print(message)


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

        parsed_args = parser.parse_args(*args)
        host = parsed_args.host
        port = int(parsed_args.port)
        game_id = parsed_args.game_id
        username = parsed_args.username
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
        self.host = host
        self.port = port
        self.game_id = game_id
        self.username = username
        self.keep_running = False
        self.auction_request_hook = auction_request_hook
        self.auction_result_hook = auction_result_hook
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def run(self):
        """
           Log into the Auction Exchange Server and respond to auction requests.
        """
        print(self.host, self.port, self.username, self.game_id)
        self.sock.connect((self.host, self.port))
        print("connected")
        print("|MT=LOGIN:GI={}:UN={}|".format(self.game_id, self.username))
        self._send_message("|MT=LOGIN:GI={}:UN={}|".format(self.game_id, self.username))

        received = ''
        self.keep_running = True
        print("loop time")
        while self.keep_running:
            try:
                received = self._run_loop(received)
            except Exception as ex:
                print(ex)
                self.keep_running = False
        self.sock.shutdown(0)

    def _run_loop(self, received):
        res = self.sock.recv(1024)
        if not res:
            raise Exception("Server socket closed")
        received += res.decode("ascii")
        print(received)
        match = self._MESSAGE_PATTERN.search(received)
        if match is None:
            return received

        message = match.group(0)
        message_type = match.group(1)
        print(message)
        self._handle_server_message(message_type, message)
        received = received[len(message):]
        return received

    def _handle_server_message(self, message_type, message):
        if message_type == "AUCTION_REQUEST":
            auction_id = int(re.search('ID=(\d*)', message).group(1))
            ht, sz = self.auction_request_hook(auction_id)
            message = "|MT=AUCTION_RESPONSE:UN={}:GI={}:ID={}:HT={}:SZ={}|".format(
                self.username, self.game_id, auction_id, ht, sz
            )
            self._send_message(message)
        elif message_type == "AUCTION_RESULT":
            self.auction_result_hook(message)
        elif message_type == "REJECT":
            print("Received reject message: " + message)
        if message_type == "END_OF_GAME":
            self.keep_running = False
            sys.exit(0)

    def _send_message(self, message):
        self.sock.sendall(bytes(message, "ascii"))

client = Client.initialize_client()
client.run()

