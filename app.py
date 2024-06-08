# app.py
import os
# import numpy as np
from flask import Flask, jsonify, request, render_template
from sample_hsi import sample_wrapper
# from omegaconf import OmegaConf
# from hydra import compose, initialize


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move_cube', methods=['POST'])
def move_cube():
    print(os.getcwd())
    data = request.json
    trajectory = data['trajectory']
    print(data)
    obj_locs = {obj_name.split('.')[0]: data[obj_name] for obj_name in data.keys() if 'trajectory' not in obj_name}

    res = sample_wrapper(trajectory, obj_locs)

    return jsonify(res)

if __name__ == '__main__':
    # os.environ["HYDRA_FULL_ERROR"] = "1"
    # initialize(version_base=None, config_path="./config")
    # OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))
    app.run(debug=True)
