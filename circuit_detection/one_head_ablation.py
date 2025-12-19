import argparse
import torch
import os
from glob import glob
import torch.nn.functional as F
import time
import json
from compute_abc_data import compute_abc_data
import math
from tqdm import tqdm
import ast
import copy
import numpy as np

##Our functions
from ioi_task import run_ioi_task,run_comparison_on_two_circuits, setup_ioi_task

def circuit_ablate_a_single_head(head):
    ablated_model = {str(i): list(range(12)) for i in range(12)}

    ablated_model[str(head[0])].remove(head[1])

    ablated_model_config = {
        "attention_heads": ablated_model,
        "ablate_mlp": False,
        "name": f"1hablation_{head}"
    }

    return ablated_model_config

def compute_one_head_ablation_experiment(args):
    np.random.seed(42)

    model, abc_data, loader, criterion, template_key = setup_ioi_task(args)

    one_head_ablation_log = {}

    complete_model_run_log = run_ioi_task(model, loader, criterion, args.device, args, args.size, template_key, circuit_name="full_model")
    one_head_ablation_log["complete_model"] = complete_model_run_log
    head_list = [(layer, head) for layer in range(12) for head in range(12)] 

    for head in head_list:

        ablation_config = circuit_ablate_a_single_head(head)

        model.ablate(abc_data, ablation_config)

        head_run_log = run_ioi_task(model, loader, criterion, args.device, args, args.size, template_key, circuit_name=ablation_config["name"])

        one_head_ablation_log[str(head)] = head_run_log

    #############################################################################################
    # Store the results
    #############################################################################################^A

    experiment_path = os.path.join(args.results_folder, f'1hablation_{args.prompt_type}_{template_key}.json')

    with open(experiment_path, "w") as f:
        json.dump(one_head_ablation_log, f, indent=4)

    print(f"Saved completeness results to {experiment_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--results_folder", type = str, default = "results/1hablation", help = "Results folder")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("-s", "--size", type = int, default = 8192, help = "Size of the IOI dataset (power of 2 is simpler for alignment with batch sizes)")
    parser.add_argument("-n", "--num_sets", type = int, default = 3, help = "Number of sets to use")
    parser.add_argument("-b", "--batch_size", type = int, default = 512, help = "Size of the batch (can be as large as vram allows as this is eval mode)")
    parser.add_argument("-t", "--num_templates", type = int, default = 1, help = "Number of different templates to use.")
    parser.add_argument("-p", "--prompt_type", type = str, default = 'BABA', help = "Template to use.")
    parser.add_argument("--configs_folder", type = str, default = "configs", help = "Configurations folder")
    parser.add_argument("--ablation_set_type", type = str, choices = ["random","set","adversarial"],default = "random", help = "THe way the sets are selected")
    parser.add_argument("--completeness_configs_folder", type = str, default = "configs/completeness_sets", help = "Configurations folder")
    parser.add_argument("--completeness_config", type = str, default = "categories", help = "Configurations folder")
    parser.add_argument("-c", "--config", type = str, default = "ioi_paper_ablation.json", help = "Ablation config file")
    parser.add_argument("-i", "--template_index", type = int, default = 0, help = "Index of the template to use (if only one template is selected).")
    parser.add_argument("--probabilities", type = float, nargs="+",default = [0.5], help = "Probability of keeping each head")


    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)
    os.makedirs(args.configs_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)

    compute_one_head_ablation_experiment(args)

