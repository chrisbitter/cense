import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import requests
import http
import time


def test_run():
    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "RUN"})
    assert r.json() == "RUN"
    assert r.status_code == http.HTTPStatus.OK

    for ii in range(5):
        time.sleep(5)
        r = requests.post("http://127.0.0.1:5000/get_statistics")

        df = pd.DataFrame(r.json())
        df.set_index("run_number")

        print(df.shape, r.status_code)

        #plt.figure()
        #df.plot(y="steps")
        #plt.figure()
        #df.plot(y="reward")
        if df.shape[0]:
            plt.figure()
            df.plot(y="exploration_probability")
            plt.show()

        if ii == 2:
            requests.post("http://127.0.0.1:5000/reset")

    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "SHUTDOWN"})
    assert r.json() == "SHUTDOWN"
    assert r.status_code == http.HTTPStatus.OK


def test_status():
    r = requests.post("http://127.0.0.1:5000/status")
    assert r.json() == "PAUSE"
    assert r.status_code == http.HTTPStatus.OK

    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "PAUSE"})
    assert r.json() == "PAUSE"
    assert r.status_code == http.HTTPStatus.OK

    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "FAIL"})
    assert r.json() == "PAUSE"
    assert r.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY


def test_shutdown():
    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "SHUTDOWN"})
    assert r.json() == "SHUTDOWN"
    assert r.status_code == http.HTTPStatus.OK


def test_params():
    r = requests.post("http://127.0.0.1:5000/get_params")
    assert r.status_code == http.HTTPStatus.OK
