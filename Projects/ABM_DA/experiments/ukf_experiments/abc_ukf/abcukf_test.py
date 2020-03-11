# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

"if running this file on its own. this will move cwd up to ukf_experiments."
if os.path.split(os.getcwd())[1] != "ukf_experiments":
    os.chdir("..")

sys.path.append("modules")
from ukf_fx import fx
from poly_functions import poly_count, grid_poly
from ukf_plots import ukf_plots
import default_ukf_configs as configs
from ukf_ex2 import hx2

sys.path.append("../../stationsim")
from stationsim_model import Model

import numpy as np





if __name__ == "__main__":
    model_params = configs.model_params
    model_params["pop_total"] = 5
    base_model = Model(**model_params)

    for _ in range(10):
        base_model.step()

    z1 = base_model.get_state(sensor = "location")

    for _ in range(5):
        base_model.step()

    z2 = base_model.get_state(sensor = "location")


