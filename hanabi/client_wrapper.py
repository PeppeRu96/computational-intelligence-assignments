import sys
from threading import Thread
import GameData
import socket
from constants import *
import os
from sys import stdout
import logging
from datetime import datetime
from enum import Enum

HANABI_FOLDER = os.path.dirname(__file__)
LOG_BASE_PATH = os.path.join(HANABI_FOLDER, "logs")
if not os.path.exists(LOG_BASE_PATH):
    os.mkdir(LOG_BASE_PATH)

class GameStatus(Enum):
    LOBBY = 1
    GAME = 2



class Client:
    def __init__(self, server_ip, server_port, player_name, player, log_file=True):
        # Server duty
        self.__server_ip = server_ip
        self.__server_port = server_port
        self.__sock = None

        # Game stuff
        self.player_name = player_name
        self.player = player
        self.player.client = self
        self.status = GameStatus.LOBBY
        self.game_count = 0
        self.ci_thread = None

        # Logger configuration
        time_and_date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        logger_name = f"log_{self.player_name}_{time_and_date_str}"
        log_fname = f"{logger_name}.log"
        log_fpath = os.path.join(LOG_BASE_PATH, log_fname)

        self.logger = logging.getLogger(logger_name)
        self.logger_fonly = logging.getLogger(logger_name + "_fileonly")
        formatter = logging.Formatter(f'%(asctime)s %(levelname)s: %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p")
        basic_formatter = logging.Formatter(f'%(message)s', datefmt="%m/%d/%Y %I:%M:%S %p")
        if log_file:
            file_handler = logging.FileHandler(filename=log_fpath)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger_fonly.addHandler(file_handler)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(basic_formatter)
        self.logger.addHandler(stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger_fonly.setLevel(logging.INFO)

    # Private utility functions
    def __connect_request(self):
        request = GameData.ClientPlayerAddData(self.player_name)
        self.__sock.connect((self.__server_ip, self.__server_port))
        self.__sock.send(request.serialize())

    def __connection_accepted(self):
        # We expect to receive only GameData.ServerPlayerConnectionOk from the server
        data = self.__receive_data()
        if type(data) is GameData.ServerPlayerConnectionOk:
            self.logger.info(f"Connection accepted by the server. Welcome {self.player_name}!")
            return True
        elif type(data) is GameData.ServerActionInvalid:
            self.logger.error(f"Server: Invalid response. Reason:\n{data.message}")
            return False
        else:
            self.logger.error(f"Unexpected server response.")
            return False

    def __receive_data(self):
        data = self.__sock.recv(DATASIZE)
        if not data:
            return None
        data = GameData.GameData.deserialize(data)

        return data

    def __send(self, data):
        self.__sock.send(data.serialize())

    def print_status(self):
        print(f"[Identity: {self.player_name} - {self.status.name}]: ", end="")
        stdout.flush()

    def connect_and_listen(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.__sock:
            # Connection request
            self.__connect_request()

            # Check if the connection has been accepted or rejected
            if not self.__connection_accepted():
                os._exit(0)

            self.player.connection_accepted()

            self.game_count += 1

            # Start a new thread to manage the console input - you may want to interrupt the AI or provide
            # manual commands in case of human player
            self.ci_thread = Thread(target=self.player.console_input)
            self.ci_thread.start()

            # Listen and dispatch the events to another thread (AI)
            while 1:
                # Receive data
                data = self.__receive_data()
                if not data:
                    continue

                if type(data) is GameData.ServerPlayerStartRequestAccepted:
                    self.logger.info(f"Ready players: {data.acceptedStartRequests}/{data.connectedPlayers}")
                elif type(data) is GameData.ServerStartGameData:
                    self.__send(GameData.ClientPlayerReadyData(self.player_name))
                    self.status = GameStatus.GAME
                    # The game could crash even now because if all the acknowledgments are not received by the server,
                    # the game does not proceed.
                elif type(data) is GameData.ServerActionInvalid:
                    self.logger.warning(f"Server replied with: Invalid action performed.\nReason: {data.message}")
                elif type(data) is GameData.ServerInvalidDataReceived:
                    self.logger.error(f"Server replied with: Invalid data received.\nData:\n{data.data}")
                elif type(data) is GameData.ServerGameOver:
                    self.game_count += 1

                # Blocking event handler (it can be an AI or a simple print) - This is blocking because
                # when it is our turn, we are not interested in handling other async commands from the server.
                # Error handling or disconnection are not handled
                self.player.event_handler(data)
                if self.status == GameStatus.LOBBY:
                    self.print_status()

    # Useful wrappers to send requests to the server
    def ready(self):
        self.__send(GameData.ClientPlayerStartRequest(self.player_name))

    def show(self):
        self.__send(GameData.ClientGetGameStateRequest(self.player_name))

    def hint(self, dest_player_name, type, value):
        self.__send(GameData.ClientHintData(self.player_name, dest_player_name, type, value))

    def discard(self, card_position):
        self.__send(GameData.ClientPlayerDiscardCardRequest(self.player_name, card_position))

    def play(self, card_position):
        self.__send(GameData.ClientPlayerPlayCardRequest(self.player_name, card_position))
