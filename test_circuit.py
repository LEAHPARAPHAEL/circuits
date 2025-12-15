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
from tqdm import tqdm

##Our functions
from ioi_task import run_ioi_task

from heads_detection import detect_heads


def test_ioi_circuit(args):

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
    # Gets or creates the result dict for the model
    # First evaluation loop for the whole model
    #############################################################################################

    log = run_ioi_task(model, loader, criterion, device, args, size, template_key,"gpt2")

    #############################################################################################
    # Builds the circuit
    #############################################################################################

    ablation_config_path = os.path.join(args.configs_folder, args.config)

    if os.path.isfile(ablation_config_path):
        ablation_config = json.load(open(ablation_config_path, "r"))
    else:
        raise(FileNotFoundError("The specified config file does not exist."))

    model.ablate(abc_data, ablation_config)


    #############################################################################################
    # Gets or creates the result dict
    # Second evaluation loop for the circuit
    #############################################################################################

    log = run_ioi_task(model, loader, criterion, device, args, size, template_key,ablation_config["name"])

    detect_heads(args, abc_data = abc_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--results_folder", type = str, default = "results", help = "Results folder")
    parser.add_argument("--detection_folder", type = str, default = "detection", help = "Detection folder")
    parser.add_argument("--plots_folder", type = str, default = "plots", help = "Plots folder")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("-s", "--size", type = int, default = 8192, help = "Size of the IOI dataset (power of 2 is simpler for alignment with batch sizes)")
    parser.add_argument("-b", "--batch_size", type = int, default = 512, help = "Size of the batch (can be as large as vram allows as this is eval mode)")
    parser.add_argument("-t", "--num_templates", type = int, default = 1, help = "Number of different templates to use.")
    parser.add_argument("-p", "--prompt_type", type = str, default = 'BABA', help = "Template to use.")
    parser.add_argument("--configs_folder", type = str, default = "configs", help = "Configurations folder")
    parser.add_argument("-c", "--config", type = str, default = "ioi_paper_ablation.json", help = "Ablation config file")
    parser.add_argument("-i", "--template_index", type = int, default = 0, help = "Index of the template to use (if only one template is selected).")


    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)
    os.makedirs(args.configs_folder, exist_ok=True)
    os.makedirs(args.detection_folder, exist_ok=True)
    os.makedirs(args.plots_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "heads"), exist_ok=True)

    test_ioi_circuit(args)


