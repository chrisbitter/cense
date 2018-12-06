import pandas as pd
import numpy as np
import torch
import requests
import http
import time


def test_status():
    r = requests.post("http://127.0.0.1:5000/status")
    assert r.json() == "PAUSE"
    assert r.status_code == http.HTTPStatus.OK

    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "RUN"})
    assert r.json() == "RUN"
    assert r.status_code == http.HTTPStatus.OK

    time.sleep(5)

    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "PAUSE"})
    assert r.json() == "PAUSE"
    assert r.status_code == http.HTTPStatus.OK

    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "FAIL"})
    assert r.json() == "PAUSE"
    assert r.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY

    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "SHUTDOWN"})
    assert r.json() == "SHUTDOWN"
    assert r.status_code == http.HTTPStatus.OK



def test_params():
    r = requests.post("http://127.0.0.1:5000/get_params")
    assert r.status_code == http.HTTPStatus.OK

