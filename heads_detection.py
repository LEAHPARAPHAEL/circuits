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
from detection_utils import get_duplicate_token_head_detection_pattern, get_induction_head_detection_pattern, get_previous_token_head_detection_pattern, compute_head_attention_similarity_score
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

def detect_heads(args):

    device = args.device

    #############################################################################################
    # Circuit config
    #############################################################################################

    ablation_config_path = os.path.join(args.configs_folder, args.config)

    if os.path.isfile(ablation_config_path):
        ablation_config = json.load(open(ablation_config_path, "r"))
    else:
        raise(FileNotFoundError("The specified config file does not exist."))


    #############################################################################################
    # Builds the model
    #############################################################################################

    start_time = time.time()
    model = AblatableGPT2Model.from_pretrained("gpt2", attn_implementation = "eager")
    print(f"Loaded GPT2 model: total time {time.time() - start_time:.2f} seconds")
    model.to(device)
    model.eval()


    #############################################################################################
    # Gets or creates the result dict for the circuit
    #############################################################################################

    detection_path = os.path.join(args.detection_folder, f'{ablation_config["name"]}.pth')

    if os.path.isfile(detection_path):
        log = torch.load(detection_path, map_location = device)
    else:
        log = {}

    #############################################################################################
    # Finds all precomputed templates and evaluates them all.
    #############################################################################################

    precomputed_templates = os.listdir(args.checkpoints_folder) 

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    for template_path in tqdm(precomputed_templates, desc = "Templates"):
        template_key = template_path.split("_")[1]
        template_index = int(template_path.split("_")[-1].split("=")[-1])

        print("----------------------------------------------------------------------------------------------------")
        print(f"Template {template_key} | Template index : {template_index}")


        if template_key not in log :
            log[template_key] = {}
        
        if str(template_index) not in log[template_key]:

            log[template_key][str(template_index)] = {}

            abc_data = torch.load(os.path.join(args.checkpoints_folder, template_path), map_location = device)

            model.ablate(abc_data, ablation_config)


            ioi_samples = IOIDataset(
                prompt_type=template_key,
                N=1,
                nb_templates=1,
                seed = 0,
                template_idx=template_index
            )



            sentence = ioi_samples.sentences[0]

            #############################################################################################
            # Builds the dataset and dataloader
            #############################################################################################

            seq_len = ioi_samples.toks.shape[1]

            print("----------------------------------------------------------------------------------------------------")
            print("Sequence length : ", seq_len)

            ioi_inputs = ioi_samples.toks.long()[:, :seq_len - 1]

            #############################################################################################
            # Computes the attention weights
            #############################################################################################

            model.start_detection()

            model(ioi_inputs.to(device), output_attentions = True)

            attention_weights = model.stop_detection()
            
            log[template_key][str(template_index)]["sentence"] = sentence
            log[template_key][str(template_index)]["tokens"] = ioi_inputs
            log[template_key][str(template_index)]["attention_weights"] = attention_weights


            #############################################################################################
            # Computes the detection patterns
            #############################################################################################
            
            previous_token_pattern = get_previous_token_head_detection_pattern(ioi_inputs).to(device)
            duplicate_token_pattern = get_duplicate_token_head_detection_pattern(ioi_inputs).to(device)
            induction_head_pattern = get_induction_head_detection_pattern(ioi_inputs).to(device)

            log[template_key][str(template_index)]["previous_token_pattern"] = previous_token_pattern
            log[template_key][str(template_index)]["duplicate_token_pattern"] = duplicate_token_pattern
            log[template_key][str(template_index)]["induction_head_pattern"] = induction_head_pattern




        else:
            previous_token_pattern = log[template_key][str(template_index)]["previous_token_pattern"].to(device) 
            duplicate_token_pattern = log[template_key][str(template_index)]["duplicate_token_pattern"].to(device) 
            induction_head_pattern = log[template_key][str(template_index)]["induction_head_pattern"].to(device) 

            sentence = log[template_key][str(template_index)]["sentence"] 
            ioi_inputs = log[template_key][str(template_index)]["tokens"].to(device) 
            attention_weights = log[template_key][str(template_index)]["attention_weights"] 

            seq_len = ioi_inputs.shape[1] + 1

        #############################################################################################
        # Computes the metrics for each head
        #############################################################################################

        heads_ablation_config = ablation_config["attention_heads"]

        patterns = [previous_token_pattern, duplicate_token_pattern, induction_head_pattern]
        
        pattern_names = ["previous_token_head", "duplicate_token_head", "induction_head"]

        tokens = tokenizer.convert_ids_to_tokens(ioi_inputs.squeeze(0))
        tokens = [token.replace("Ġ", "") for token in tokens]

        for layer_idx, heads_indices in heads_ablation_config.items():

            layer_weights = attention_weights[str(layer_idx)].to(device)

            for head_idx in heads_indices:

                head_weights = layer_weights[:, head_idx, :seq_len-1, :seq_len-1]

                head_key = f"{layer_idx}-{head_idx}"

                for pattern, pattern_name in zip(patterns, pattern_names):

                    #mul = compute_head_attention_similarity_score(head_weights, pattern, error_measure = "mul")
                    abs = compute_head_attention_similarity_score(head_weights, pattern, error_measure = "abs", exclude_bos=False, exclude_current_token=False)

                    if abs >= args.threshold:
                        print(f"Head ({head_key}) | {pattern_name} : {abs}")
                        plot_pattern(args, tokenizer, ablation_config["name"], head_key, head_weights, 
                                     pattern, pattern_name, tokens)

                
    torch.save(log, detection_path)


def plot_pattern(args, tokenizer, circuit_name, head_key, head_weights, pattern, pattern_name, tokens):

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 7))

    axes[0].imshow(head_weights.squeeze(0).cpu(), cmap="coolwarm", aspect="auto")
    axes[0].set_yticks(
        range(len(tokens)), tokens
    )
    axes[0].set_xticks(
        range(len(tokens)), tokens, rotation=90
    )
    axes[0].set_title("Attention weights")

    im = axes[1].imshow(pattern.cpu(), cmap="coolwarm", aspect="auto")
    axes[1].set_yticks(
        range(len(tokens)), tokens
    )
    axes[1].set_xticks(
        range(len(tokens)), tokens, rotation=90
    )
    axes[1].set_title(f"Pattern : {pattern_name}")  

    fig.colorbar(im, fraction=0.046, pad=0.04)
    #fig.colorbar()
    plt.tight_layout()

    plt.savefig(os.path.join(args.plots_folder, f"{circuit_name}_{pattern_name}_({head_key})"))

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--detection_folder", type = str, default = "detection", help = "Detection folder")
    parser.add_argument("--plots_folder", type = str, default = "plots", help = "Plots folder")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("--configs_folder", type = str, default = "configs", help = "Configurations folder")
    parser.add_argument("-c", "--config", type = str, default = "ioi.json", help = "Ablation config file")
    parser.add_argument("-t", "--threshold", type = float, default = 0.93, help = "Threshold")

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)
    os.makedirs(args.configs_folder, exist_ok=True)
    os.makedirs(args.detection_folder, exist_ok=True)
    os.makedirs(args.plots_folder, exist_ok=True)

    detect_heads(args)


