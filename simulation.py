import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import json
import os

from SATh_utilities import *

def param_title(dict_, *args):
#This function allows us to print the parameters used for the simulation as the title of a plot.
    for tag in args:
        del dict_[tag]
    plt.title(dict_)

def compare_errors(u1,u2):
#A function to compare the norm error of two functions
    return np.linalg.norm(u1-u2)


def json_todict(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: '{json_file_path}' does not contain valid JSON data.")

def plot_problem(p):
    s = Problem(**p)
    fig, ax = plt.subplots()
    X = s.mesh.nodes

    ax.plot(X, s.funcs.init_sol, "--", label="initial")
    ax.plot(X, s.sol_ex(), ".-.", label="exact")
    for theta in np.arange(p["th_min"]*10, p["th_max"]*10)/10:  #start at th_min= 0.2 or 0.3 if you give a bad CFL
        p["theta"] = theta
        s = Problem(**p)
        ax.plot(X, s.sol_num, label=f"{theta}")

    plt.legend()
    param_title(parameters, "a","b","theta","params","init_func")

    save_path = p["Path"] + f"/plot_{p['cfl']}_{p['Nx']}_{p['tf']}.png"
    print(save_path)
    os.open(save_path, os.O_CREAT | os.O_TRUNC, 0o666)
    plt.savefig(save_path)


if __name__ == "__main__":
    json_file_path = "args.json"  #input("Enter the path to the .JSON file: ")
    parameters = json_todict(json_file_path)
    print(parameters)
    plot_problem(parameters)
