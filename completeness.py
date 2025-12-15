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
import numpy as np

##Our functions
from ioi_task import run_ioi_task


def run_comparison_task(ablated_model_config, ablated_circuit_config, model, abc_data, loader, criterion, device, args, size, template_key,circuit_name):

    #############################################################################################
    # Run the IOI task on the ablated model 
    #############################################################################################^A

    model.ablate(abc_data, ablated_model_config)
    ablated_model_log = run_ioi_task(model, loader, criterion, device, args, size, template_key, circuit_name=ablated_model_config["name"])

    #############################################################################################
    # Run the IOI task on the ablated circuit 
    #############################################################################################^A

    model.ablate(abc_data, ablated_circuit_config)
    ablated_circuit_log = run_ioi_task(model, loader, criterion, device, args, size, template_key, circuit_name=ablated_circuit_config["name"])

    #############################################################################################
    # Store the results of this random ablation
    #############################################################################################^A

    completeness_score = ablated_model_log[args.prompt_type][template_key]["Logit difference"] - ablated_circuit_log[args.prompt_type][template_key]["Logit difference"]
    exp_results = {
        "ablated_model_difference": ablated_model_log[args.prompt_type][template_key]["Logit difference"],
        "ablated_circuit_difference": ablated_circuit_log[args.prompt_type][template_key]["Logit difference"],
        "completeness_score": completeness_score
    }

    return exp_results



def make_ablation_configs(circuit_config,set_config,set_name):
    ablated_circuit = copy.deepcopy(circuit_config["attention_heads"])
    ablated_model = {str(i): list(range(12)) for i in range(12)}

    ##set config is a list of lists of 2 elements
    for head in set_config:
        ablated_circuit[str(head[0])].remove(head[1])
        ablated_model[str(head[0])].remove(head[1])
    name = circuit_config['name']
    ablated_circuit_config = {
        "attention_heads": ablated_circuit,
        "ablate_mlp": circuit_config["ablate_mlp"],
        "name": f"{name}_circuit_ablated_set={set_name}"
    }

    ablated_model_config = {
        "attention_heads": ablated_model,
        "ablate_mlp": circuit_config["ablate_mlp"],
        "name": f"{name}_model_ablated_set={set_name}"
    }

    return ablated_circuit_config, ablated_model_config

def generate_random_ablation_set(ablation_config, p):
    random_circuit = copy.deepcopy(ablation_config["attention_heads"])
    model_with_random_ablation = {str(i): list(range(12)) for i in range(12)}
    kept_heads = []
    signature = ""
    ablated_heads = []
    for layer in random_circuit.keys():
        for head in random_circuit[layer]:
            keep = np.random.binomial(n=1, p = p)
            signature += str(keep)
            if keep == 0:
                ablated_heads.append((int(layer), head))
                model_with_random_ablation[str(layer)].remove(head)  
                random_circuit[layer].remove(head)
            else:
                kept_heads.append((int(layer), head))

    ##The random config is caracterized by it's signature
    ##0's when the head is removed, 1's when it is kept
    ## the order is the one when reading the circuit config in order
    name = ablation_config["name"]
    random_config = {
        "attention_heads": random_circuit,
        "ablate_mlp": ablation_config["ablate_mlp"],
        "name": f"{name}_circuit_ablated_{signature}"
    }
    random_ablated_model_config = {
        "attention_heads": model_with_random_ablation,
        "ablate_mlp": ablation_config["ablate_mlp"],
        "name": f"{name}_model_ablated_{signature}"
    }
    return random_config, random_ablated_model_config,signature, kept_heads , ablated_heads

def compute_completeness(args):
    np.random.seed(42)

    size = args.size
    device = args.device

    #############################################################################################
    # Builds the IOI dataset
    #############################################################################################

    ioi_samples = IOIDataset(
        prompt_type=args.prompt_type,
        N=size,
        nb_templates=args.num_templates,
        seed = 0,
        template_idx=args.template_index
    )

    print("----------------------------------------------------------------------------------------------------")
    print("A few samples from the IOI dataset  : ")
    for sentence in ioi_samples.sentences[:5]:
        print(sentence)
    print("----------------------------------------------------------------------------------------------------")


    #############################################################################################
    # Loads the sums over the ABC dataset.
    #############################################################################################

    abc_path = os.path.join(args.checkpoints_folder, f"abc_{args.prompt_type}_")
    template_key = f"T={args.template_index}" if args.num_templates == 1 else f"N={args.num_templates}"
    abc_path += template_key

    template_O_position = ioi_samples.O_position

    if not os.path.isfile(abc_path):
        print("----------------------------------------------------------------------------------------------------")
        print("ABC data not found. Fall back to full computation.")
        compute_abc_data(args)

    abc_data = torch.load(abc_path, map_location = device)
    print(f"Successfully loaded previous ABC means from {abc_path}")


    #############################################################################################
    # Builds the dataset and dataloader
    #############################################################################################

    seq_len = ioi_samples.toks.shape[1]

    print("----------------------------------------------------------------------------------------------------")
    print("Sequence length : ", seq_len)

    ioi_inputs = ioi_samples.toks.long()[:, :seq_len - 1]

    ioi_labels = ioi_samples.toks.long()[:, seq_len - 1]
    ioi_O_labels = ioi_samples.toks.long()[:, template_O_position] ##This is only compatible with a single template running

    ioi_dataset = TensorDataset(ioi_inputs, ioi_labels, ioi_O_labels)

    loader = DataLoader(ioi_dataset, batch_size = args.batch_size, shuffle = False)

    #############################################################################################
    # Builds the model
    #############################################################################################^
    start_time = time.time()
    model = AblatableGPT2Model.from_pretrained("gpt2")
    print(f"Loaded GPT2 model: total time {time.time() - start_time:.2f} seconds")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction = "sum")

    #############################################################################################
    # Load the circuit config
    #############################################################################################^

    ablation_config_path = os.path.join(args.configs_folder, args.config)

    if os.path.isfile(ablation_config_path):
        ablation_config = json.load(open(ablation_config_path, "r"))
    else:
        raise(FileNotFoundError("The specified config file does not exist."))


    completeness_log = {}

    if args.ablation_set_type == "random":

        set_key = f"num_runs={args.num_sets}_prob={args.probability}"
        current_biggest_difference = -math.inf

        for n in range(args.num_sets):

            #############################################################################################
            # Generate random abltation set and the corresponding model/circuit
            #############################################################################################^

            random_circuit_config, random_ablated_model_config,signature, kept_heads, ablated_heads = generate_random_ablation_set(ablation_config,p = args.probability)
            print(f'Generating random ablation set {n+1} / {args.num_sets}')
            print(f"We abalted the following head {ablated_heads}")

            #############################################################################################
            # Run comparative exps
            #############################################################################################^

            exp_results = run_comparison_task(random_ablated_model_config, random_circuit_config, model, abc_data, loader, criterion, device, args, size, template_key,circuit_name=ablation_config["name"])
            completeness_score = exp_results["completeness_score"]

            completeness_log[signature] = exp_results
            current_biggest_difference = max(current_biggest_difference ,completeness_score)
            print(f"Completeness score for random set {n+1} / {args.num_sets} : {completeness_score:.4f} minimal score : {current_biggest_difference}")

            completeness_log["largest difference found"] =  current_biggest_difference

    elif args.ablation_set_type == "set":
        set_key = args.completeness_config
        #############################################################################################
        # Loads the completeness config
        #############################################################################################
        completeness_config_path = os.path.join(args.completeness_configs_folder, f'{args.completeness_config}.json')
        completeness_config = json.load(open(completeness_config_path, "r"))

        for set_name in completeness_config.keys():
            set_config = completeness_config[set_name]

            ablated_circuit_config, ablated_model_config = make_ablation_configs(ablation_config,set_config,set_name)
            exp_results = run_comparison_task(ablated_model_config, ablated_circuit_config, model, abc_data, loader, criterion, device, args, size, template_key,circuit_name=ablation_config["name"])
            completeness_log[set_name] = exp_results

    #############################################################################################
    # Store the results
    #############################################################################################^A

    completeness_log["name"] = ablation_config["name"]
    completeness_log["ablation_set_type"] = args.ablation_set_type

    completeness_results_path = os.path.join(args.results_folder, f'completeness_{args.prompt_type}_{template_key}_type={args.ablation_set_type}_{set_key}.json')
    with open(completeness_results_path, "w") as f:
        json.dump(completeness_log, f, indent=4)
    print(f"Saved completeness results to {completeness_results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--results_folder", type = str, default = "results/completeness", help = "Results folder")
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

    for p in args.probabilities:
        args.probability = p
        compute_completeness(args)


