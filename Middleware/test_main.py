import pandas as pd
import numpy as np
import torch
import requests
import http


def test_status():
    r = requests.post("http://127.0.0.1:5000/status")
    assert r.json() == "STOP"
    assert r.status_code == http.HTTPStatus.OK

    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "RUN"})
    assert r.json() == "RUN"
    assert r.status_code == http.HTTPStatus.OK

    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "PAUSE"})
    assert r.json() == "PAUSE"
    assert r.status_code == http.HTTPStatus.OK

    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "STOP"})
    assert r.json() == "STOP"
    assert r.status_code == http.HTTPStatus.OK

    r = requests.post("http://127.0.0.1:5000/status", data={'new_status': "FAIL"})
    assert r.json() == "STOP"
    assert r.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY


def test_params():
    r = requests.post("http://127.0.0.1:5000/get_params")
    print(r.json())
