from model import AblatableGPT2Model
import argparse
import torch
from ioi_dataset import IOIDataset
import os
from torch.utils.data import TensorDataset, DataLoader
from glob import glob
import torch.nn as nn
import torch.nn.functional as F


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
    )

    print("----------------------------------------------------------------------------------------------------")
    print("A few samples from the IOI dataset  : ")
    for sentence in ioi_samples.sentences[:5]:
        print(sentence)
    print("----------------------------------------------------------------------------------------------------")

    #############################################################################################
    # Loads the sums over the ABC dataset.
    #############################################################################################

    potential_filepaths = glob(f"{args.abc_data_path}_*", root_dir = args.checkpoints_folder)

    if potential_filepaths:
        abc_data_path = os.path.join(args.checkpoints_folder, potential_filepaths[0])
        abc_data = torch.load(abc_data_path, map_location = device)

        print(f"Successfully loaded previous ABC means from {abc_data_path}")

        last_sample_index = abc_data["last_sample_index"]

    else:
        raise(FileNotFoundError("No ABC data file was found ! One is required to perform ablation. Please execute" \
        "the file compute_abc_data.py first. "))


    #############################################################################################
    # Builds the dataset and dataloader
    #############################################################################################

    seq_len = ioi_samples.toks.shape[1]

    ioi_inputs = ioi_samples.toks.long()[:, :seq_len - 1]

    ioi_labels = ioi_samples.toks.long()[:, seq_len - 1]

    ioi_dataset = TensorDataset(ioi_inputs, ioi_labels)

    loader = DataLoader(ioi_dataset, batch_size = args.batch_size, shuffle = False)


    #############################################################################################
    # Builds the model
    #############################################################################################^

    model = AblatableGPT2Model.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()


    #############################################################################################
    # First evaluation loop for the whole model
    #############################################################################################

    print("----------------------------------------------------------------------------------------------------")
    print(f"Evaluating the entire GPT2 model on the IOI dataset with template {args.prompt_type} and {size} samples.")

    total_loss = 0
    for batch_idx, (batch_inputs, batch_labels) in enumerate(loader):

        with torch.no_grad():

            inputs = batch_inputs.to(device)
            labels = batch_labels.to(device)
            
            outputs = model(inputs)

            all_logits = outputs.logits

            next_token_logits = all_logits[:, -1, :]

            loss = criterion(next_token_logits, labels)

            total_loss += loss.item()

            if ((batch_idx + 1) * args.batch_size) % args.test_steps == 0:
                print(f"Average loss for the entire model after {(batch_idx + 1) * args.batch_size} samples : {total_loss / (batch_idx + 1):.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Average loss for the entire model after {len(loader) * args.batch_size} samples : {avg_loss:.4f}")


    #############################################################################################
    # Second evaluation loop for the circuit
    #############################################################################################

    ablation_config_path = os.path.join(args.configs_folder, args.config)

    model.ablate(abc_data_path, ablation_config_path)

    print("----------------------------------------------------------------------------------------------------")
    print(f"Evaluating the IOI circuit on the IOI dataset with template {args.prompt_type} and {size} samples.")

    total_loss = 0
    for batch_idx, (batch_inputs, batch_labels) in enumerate(loader):

        with torch.no_grad():

            inputs = batch_inputs.to(device)
            labels = batch_labels.to(device)
            
            outputs = model(inputs)

            all_logits = outputs.logits

            next_token_logits = all_logits[:, -1, :]

            loss = criterion(next_token_logits, labels)

            total_loss += loss.item()

            if ((batch_idx + 1) * args.batch_size) % args.test_steps == 0:
                print(f"Average loss for the IOI circuit after {(batch_idx + 1) * args.batch_size} samples : {total_loss / (batch_idx + 1):.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Average loss for the IOI circuit after {len(loader) * args.batch_size} samples : {avg_loss:.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--results_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("--results_file", type = str, default = "metrics.json", help = "Checkpoints folder")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("-a", "--abc_data_path", type = str, default = "abc_data", help = "Path to the saved sums over the ABC dataset")
    parser.add_argument("-s", "--size", type = int, default = 8192, help = "Size of the IOI dataset (power of 2 is simpler for alignment with batch sizes)")
    parser.add_argument("-b", "--batch_size", type = int, default = 512, help = "Size of the batch (can be as large as vram allows as this is eval mode)")
    parser.add_argument("-t", "--num_templates", type = int, default = 30, help = "Number of different templates to use.")
    parser.add_argument("-p", "--prompt_type", type = str, default = 'mixed', help = "Template to use.")
    parser.add_argument("--configs_folder", type = str, default = "configs", help = "Configurations folder")
    parser.add_argument("-c", "--config", type = str, default = "ioi.json", help = "Ablation config file")
    parser.add_argument("--test_steps", type = int, default = 1024, help = "Number of samples between each print in the console.")


    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)
    os.makedirs(args.configs_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)

    test_ioi_circuit(args)


