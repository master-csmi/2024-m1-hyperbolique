import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import argparse
import json
import os

from Library.SATh import *

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

    if p["Scheme"] == "Theta":
        ax.plot(X, s.funcs.init_sol, "--", label="initial")
        ax.plot(X, s.sol_ex(), ".-.", label="exact")
        for theta in np.arange(p["Theta_min"]*10, p["Theta_max"]*10)/10:  #start at th_min= 0.2 or 0.3 if you give a bad CFL
            p["Theta_st"] = theta
            s = Problem(**p)
            ax.plot(X, s.sol_num, label=f"{theta}")
        param_title(p, "a","b","Theta_st","params","init_func")

    else:
        ax.plot(X, s.funcs.init_sol, "--", label="initial")
        ax.plot(X, s.sol_ex(), ".-.", label="exact")
        ax.plot(X, s.sol_num, label= "sol_num")
        #plt.title(f"Theta_st: {p["Theta_st"]} ; Theta_min: {p["Theta_min"]}")

    plt.legend()

    save_path = p["Path"] + f"/plot_{p["Problem"]}_{p["Flux"]}_{p["Theta_choice_method"]}_cfl{p['cfl']}_Nx{p['Nx']}_tf{p['tf']}.png"
    print(save_path)
    os.open(save_path, os.O_CREAT | os.O_TRUNC, 0o666)
    plt.savefig(save_path)

def main(option):
    json_file_path = "args.json"  #input("Enter the path to the .JSON file: ")
    parameters = json_todict(json_file_path)
    if option == "plot":
        print(parameters)
        plot_problem(parameters)
    else:
        print(parameters)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with options.")
    parser.add_argument("option", choices=["plot", "no_plot"], help="Option to run the simulation.")
    args = parser.parse_args()
    main(args.option)
