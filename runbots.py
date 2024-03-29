import argparse
import socket
import sys
import os
import uuid
from subprocess import Popen, PIPE

bot_scripts = ["simple_mle_estimation.py", "random_bot.py", "follow_wins.py", "switch_strat.py", "mle_estimation.py", "big_brain_estimation.py", "anti_mle_estimation.py", "malicious_bot.py", "agg_bot.py"]
# bot_scripts = ["random_bot.py"]
# bot_scripts = ["random_bot.py" for _ in range(10)]

parser = argparse.ArgumentParser()
parser.add_argument("host")
parser.add_argument("port")
parser.add_argument("-gi", "--game_id", required=True)

parsed_args = parser.parse_args()
host = parsed_args.host
port = int(parsed_args.port)
game_id = parsed_args.game_id

processes = []
os.mkdir(f"./logs/{game_id}")
for bot in bot_scripts:
    un = bot.split(".")[0] + str(uuid.uuid4())[0:5]
    log_file = un + ".log"

    process = Popen(['python', bot, host, str(port), "-gi", str(game_id), "-un", un, "-o", log_file], stdout=PIPE, stderr=PIPE)
    processes.append(process)
    #stdout, stderr = process.communicate()

exit_codes = [p.wait() for p in processes]
print(exit_codes)