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

import math


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


@app.route('/get_statistics', methods=['POST'])
def get_statistics():
    global statistics

    payload = {}

    payload["run_number"] = statistics.index.tolist()

    for col in statistics.columns:
        payload[col] = statistics[col].tolist()

    return jsonify(payload)


def exponential_decay(params):
    return lambda x: max(params["cutoff"], params["start"] * math.exp(-params["decay"] * x))


@app.route('/reset', methods=['POST'])
def reset():
    global params
    with open("default.json") as json_data:
        params = json.load(json_data)

    global run_number
    run_number = 0

    global statistics
    statistics = pd.DataFrame(columns=["steps", "reward", "exploration_probability"])

    #global running_status
    #running_status = RunningStatus.PAUSE

    global exploration_probability_function

    if params["exploration_probability"]["type"] == "exp_decay":
        exploration_probability_function = exponential_decay(params["exploration_probability"]["params"])

    else:
        raise NotImplementedError("Parameter exploration_probability wrongly configured")

    return "", http.HTTPStatus.NO_CONTENT

def run_until_terminal(exploration_probability):
    # print("run_until_terminal(exploration_probability={})".format(exploration_probability))

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

            step_reward = 2 * np.random.random() - 1

            terminal = np.random.random() > .8
            steps += 1
            reward += step_reward


        elif running_status == RunningStatus.PAUSE:
            time.sleep(.1)

        elif running_status == RunningStatus.SHUTDOWN:
            break

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
    exploration_probability_function = None
    exploration_probability = None

    app_thread = Thread(target=app.run, args=(args.host, args.port), daemon=True)
    app_thread.start()

    reset()

    while running_status != RunningStatus.SHUTDOWN:

        if running_status == RunningStatus.RUN:

            run_number += 1

            print(run_number)

            if run_number % params['runs_before_testing_from_start'] == 0:
                # todo: reset world
                exploration_probability = 0

            elif run_number % params['runs_before_advancing_start'] == 0:
                exploration_probability = 0

            else:
                exploration_probability = exploration_probability_function(run_number)

            time.sleep(.2)

            steps, reward = run_until_terminal(exploration_probability)

            statistics = statistics.append(pd.DataFrame(
                data={"steps": steps, "reward": reward, "exploration_probability": exploration_probability},
                index=[run_number]))

        else:
            time.sleep(.1)
