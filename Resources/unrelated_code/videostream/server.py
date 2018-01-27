from flask import Flask, render_template, Response
from flask.json import jsonify
from camera import Camera
import time
import json
import argparse

parser = argparse.ArgumentParser(description='Stream Camera Images.')
parser.add_argument('-D', '--dim', type=int, nargs='+', default=[40, 40, 3])
parser.add_argument('-H', '--host', default='0.0.0.0')
parser.add_argument('-P', '--port', default='5000')

args = parser.parse_args()

dim = tuple(args.dim)

cam = Camera(dim)
app = Flask(__name__)

@app.route('/')
def index():

    return cam.get_frame()

app.run(host=args.host, port=args.port)
