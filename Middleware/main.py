import http

from flask import Flask, render_template, Response
from flask.json import jsonify
import time
import json
import argparse
from threading import Thread
from flask import request
from enum import Enum
import pandas as pd
import numpy as np

class RunningStatus(Enum):
    RUN = 0
    PAUSE = 1
    SHUTDOWN = 2


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("frontend.html")


@app.route('/status', methods=['POST'])
def status():
    http_status = http.HTTPStatus.OK

    global running_status

    if 'new_status' in request.form:
        if not hasattr(RunningStatus, request.form['new_status']):
            http_status = http.HTTPStatus.UNPROCESSABLE_ENTITY
        else:
            running_status = RunningStatus[request.form['new_status']]

    return jsonify(running_status.name), http_status


@app.route('/get_params', methods=['POST'])
def get_params():
    global params
    return jsonify(params)

@app.route('/reset', methods=['POST'])
def reset():
    global params
    with open("default.json") as json_data:
        params = json.load(json_data)

    global run_number
    run_number = 0

    global statistics
    statistics = pd.DataFrame(columns=["steps", "reward", "exploration_probability"])


def run_until_terminal(exploration_probability):

    global running_status

    steps = 0
    reward = 0

    terminal = False

    while not terminal:

        if running_status == RunningStatus.RUN:
            # todo trigger world to observe state
            # todo get action from agent
            # todo noise action
            # todo trigger world to execute
            # todo get reward and terminal

            step_reward = .7

            terminal = True
            steps += 1
            reward += step_reward

            pass

    return steps, reward

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', default='127.0.0.1')
    parser.add_argument('-P', '--port', default='5000')

    args = parser.parse_args()

    running_status = RunningStatus.PAUSE

    params = None
    run_number = None
    statistics = None

    app_thread = Thread(target=app.run, args=(args.host, args.port), daemon=True)
    app_thread.start()

    reset()

    while running_status != RunningStatus.SHUTDOWN:

        if running_status == RunningStatus.RUN:

            if run_number % params.runs_before_testing_from_start == 0:

                # todo: reset world

                steps, reward = run_until_terminal(0)

                continue



            print(run_number)
            time.sleep(1)
            run_number += 1

        else:
            time.sleep(.1)
