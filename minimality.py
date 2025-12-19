from model import AblatableGPT2Model
import argparse
import torch
from ioi_dataset import IOIDataset
import os
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from compute_abc_data import compute_abc_data
import math
from tqdm import tqdm
import ast
import copy

##Our functions
from ioi_task import run_ioi_task, setup_ioi_task, run_comparison_on_two_circuits


def config_compatibility(minimality_sets_config,circuit):
    
    minimality_sets = minimality_sets_config["minimality_sets"]
    heads = [ast.literal_eval(head) for head in minimality_sets.keys()]
    for head in heads:
        if head[1] not in circuit[str(head[0])]:
            return False

    for layer in circuit.keys():
        for head in circuit[layer]:
            if (int(layer),head) not in heads:
                return False

    return True

def build_minimality_ablation_configs(circuit_config,minimality_set,head_to_evaluate):
    head_to_evaluate = ast.literal_eval(head_to_evaluate)
    circuit_config_with = copy.deepcopy(circuit_config["attention_heads"])

    for head in minimality_set:
        circuit_config_with[str(head[0])].remove(head[1])

    circuit_config_without = copy.deepcopy(circuit_config_with)
    circuit_config_without[str(head_to_evaluate[0])].remove(head_to_evaluate[1])

    print(f"for head {head_to_evaluate}, \n minimality set without heads: {circuit_config_without} \n minimality set with heads: {circuit_config_with}")
    ablation_config_with = {
        "name" : f"{circuit_config['name']}_{head_to_evaluate}_minimality_set_with",
        "attention_heads" : circuit_config_with,
        "ablate_mlp" : False
    }

    ablation_config_without = {
        "name" : f"{circuit_config['name']}_{head_to_evaluate}_minimality_set_without",
        "attention_heads" : circuit_config_without,
        "ablate_mlp" : False
    }

    return ablation_config_with, ablation_config_without

def compute_minimality(args):

    model, abc_data, loader, criterion, template_key = setup_ioi_task(args)

    #############################################################################################
    # Loads the ablation config
    #############################################################################################

    circuit_config_path = os.path.join(args.configs_folder, args.config)

    if os.path.isfile(circuit_config_path):
        circuit_config = json.load(open(circuit_config_path, "r"))
    else:
        raise(FileNotFoundError("The specified config file does not exist."))


    minimality_log = {}

    #############################################################################################
    # Loads the minimality sets config
    #############################################################################################

    minimality_sets_config_path = os.path.join(args.minimality_sets_configs_folder, args.config)
    if os.path.isfile(minimality_sets_config_path):
        minimality_sets_config = json.load(open(minimality_sets_config_path, "r"))
    else:
        raise(FileNotFoundError(f"The specified minimality config file does not exist {minimality_sets_config_path}."))

    try:
        assert config_compatibility(minimality_sets_config,circuit_config["attention_heads"])
    except:
        raise(ValueError("The minimality sets config is not compatible with the ablation config."))

    A = math.inf
    minimality_sets = minimality_sets_config["minimality_sets"]
    for head_to_evaluate in minimality_sets.keys():
        print(f'Dealing with head {head_to_evaluate} for minimality computation.')
        #############################################################################################
        # Builds the circuit
        #############################################################################################

        ablation_config_with, ablation_config_without = build_minimality_ablation_configs(circuit_config,minimality_sets[head_to_evaluate],head_to_evaluate)

        log_with,log_without = run_comparison_on_two_circuits(ablation_config_with,ablation_config_without, model, abc_data, loader, criterion, args.device, args, args.size, template_key,circuit_name=circuit_config["name"])
#        model.ablate(abc_data, ablation_config_without)
        #minimality_circuit_name = ablation_config_without["name"] + "_" + head_to_evaluate + "_minimality_set_without"
        #log_without = run_ioi_task(model, loader, criterion, args.device, args, args.size, template_key,minimality_circuit_name)

        #model.ablate(abc_data, ablation_config_with)
        #minimality_circuit_name = ablation_config_with["name"] + "_" + head_to_evaluate + "_minimality_set_with"
        #log_with = run_ioi_task(model, loader, criterion, args.device, args, args.size, template_key,minimality_circuit_name)

        difference_without = log_without[args.prompt_type][template_key]["Logit difference"]
        difference_with = log_with[args.prompt_type][template_key]["Logit difference"]
        difference_in_difference = difference_with - difference_without
        print("Minimality results for head ", head_to_evaluate)
        print("Without head : ", difference_without)
        print("With head : ", difference_with)
        print("Difference in logit difference : ", difference_in_difference)

        A = min(A, difference_in_difference)
        print("A so far : ", A)

        minimality_log[head_to_evaluate] = {
            "without_head_log" : log_without,
            "with_head_log" : log_with,
            "Difference in logit difference" : difference_in_difference
        }
    # Saves the minimality results
    minimality_results_path = os.path.join(args.results_folder, f'minimality_{args.prompt_type}_{template_key}.json')
    with open(minimality_results_path, "w") as f:
        json.dump(minimality_log, f, indent=4)
    print(f"Saved minimality results to {minimality_results_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--results_folder", type = str, default = "results/minimality", help = "Results folder")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("-s", "--size", type = int, default = 8192, help = "Size of the IOI dataset (power of 2 is simpler for alignment with batch sizes)")
    parser.add_argument("-b", "--batch_size", type = int, default = 512, help = "Size of the batch (can be as large as vram allows as this is eval mode)")
    parser.add_argument("-t", "--template_keys", type = int, nargs="+", default = [0], help = "Number of different templates to use.")
    parser.add_argument("-p", "--prompt_type", type = str, default = 'BABA', help = "Template to use.")
    parser.add_argument("--configs_folder", type = str, default = "configs", help = "Configurations folder")
    parser.add_argument("--minimality_sets_configs_folder", type = str, default = "configs/minimality_sets", help = "Configurations folder")
    parser.add_argument("-c", "--config", type = str, default = "ioi_paper_ablation.json", help = "Ablation config file")


    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)
    os.makedirs(args.configs_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)

    args.num_templates = len(args.template_keys)
    for template_key in args.template_keys:
        args.template_index = template_key
        print(f"Computing minimality for template key {template_key}")
        compute_minimality(args)