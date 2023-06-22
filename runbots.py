import argparse
import socket
import sys
import os
import uuid
from subprocess import Popen, PIPE

bot_scripts = ["samplecode.py","samplecode.py","samplecode.py"]

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