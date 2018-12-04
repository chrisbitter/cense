import http

from flask import Flask, render_template, Response
from flask.json import jsonify
import time
import json
import argparse
from threading import Thread
from flask import request
from enum import Enum


class RunningStatus(Enum):
    RUN = 0
    PAUSE = 1
    STOP = 2



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template("frontend.html")

    # return jsonify(params)


@app.route('/status', methods=['POST'])
def change_status():
    print(request.form)
    print(request.values)

    if 'new_status' not in request.args:
        return "Argument 'new_status' missing", http.HTTPStatus.UNPROCESSABLE_ENTITY

    if not hasattr(RunningStatus, request.args['new_status']):
        return "Argument 'new_status' has unknown value: " + request.args[
            'new_status'], http.HTTPStatus.UNPROCESSABLE_ENTITY

    global running_status
    running_status = RunningStatus[request.args['new_status']]

    return '', http.HTTPStatus.OK


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host', default='127.0.0.1')
    parser.add_argument('-P', '--port', default='5000')

    args = parser.parse_args()

    running_status = RunningStatus.STOP

    params = {"a": 1, "b": "2"}

    # todo: load default parameters.json in params

    Thread(target=app.run, args=(args.host, args.port)).start()

    run = 0

    while True:

        if running_status == RunningStatus.RUN:

            print(run)
            time.sleep(1)
            run += 1

        else:
            time.sleep(.1)
