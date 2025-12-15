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
from detection_utils import get_duplicate_token_head_detection_pattern, \
    get_induction_head_detection_pattern, get_previous_token_head_detection_pattern, \
        get_s_inhibition_head_detection_pattern, compute_head_attention_similarity_score, \
        get_name_mover_head_detection_pattern
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


    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("----------------------------------------------------------------------------------------------------")
    print(f"Template {args.prompt_type} | Template index : {args.template_index}")


    if args.prompt_type not in log :
        log[args.prompt_type] = {}
    
    if str(args.template_index) not in log[args.prompt_type]:


        log[args.prompt_type][str(args.template_index)] = {}

        #############################################################################################
        # Loads or computes the ABC means
        #############################################################################################

        abc_path = os.path.join(args.checkpoints_folder, f"abc_{args.prompt_type}_T={args.template_index}")

        if not os.path.isfile(abc_path):
            print("----------------------------------------------------------------------------------------------------")
            print("ABC data not found. Fall back to full computation.")
            compute_abc_data(args)

        abc_data = torch.load(abc_path, map_location = device)
        print(f"Successfully loaded previous ABC means from {abc_path}")


        model.ablate(abc_data, ablation_config)


        ioi_samples = IOIDataset(
            prompt_type=args.prompt_type,
            N=1,
            nb_templates=1,
            seed = 0,
            template_idx=args.template_index
        )

        template_O_position = ioi_samples.O_position

        sentence = ioi_samples.sentences[0]

        #############################################################################################
        # Builds the dataset
        #############################################################################################

        seq_len = ioi_samples.toks.shape[1]

        print("----------------------------------------------------------------------------------------------------")
        print("Sequence length : ", seq_len)


        ioi_inputs = ioi_samples.toks.long()[:, :seq_len - 1]
        ioi_label = ioi_samples.toks.long()[:, seq_len - 1].item()

        #############################################################################################
        # Computes the attention weights
        #############################################################################################

        model.start_detection()

        model(ioi_inputs.to(device), output_attentions = True)

        attention_weights = model.stop_detection()
        
        log[args.prompt_type][str(args.template_index)]["sentence"] = sentence
        log[args.prompt_type][str(args.template_index)]["template_O_position"] = template_O_position
        log[args.prompt_type][str(args.template_index)]["ioi_label"] = ioi_label
        log[args.prompt_type][str(args.template_index)]["tokens"] = ioi_inputs
        log[args.prompt_type][str(args.template_index)]["attention_weights"] = attention_weights


        #############################################################################################
        # Computes the detection patterns
        #############################################################################################
        
        previous_token_pattern = get_previous_token_head_detection_pattern(ioi_inputs).to(device)
        duplicate_token_pattern = get_duplicate_token_head_detection_pattern(ioi_inputs).to(device)
        induction_head_pattern = get_induction_head_detection_pattern(ioi_inputs).to(device)
        s_inhibition_head_pattern = get_s_inhibition_head_detection_pattern(ioi_inputs, template_O_position).to(device)
        name_mover_head_pattern = get_name_mover_head_detection_pattern(ioi_inputs, ioi_label).to(device)

        log[args.prompt_type][str(args.template_index)]["previous_token_pattern"] = previous_token_pattern
        log[args.prompt_type][str(args.template_index)]["duplicate_token_pattern"] = duplicate_token_pattern
        log[args.prompt_type][str(args.template_index)]["induction_head_pattern"] = induction_head_pattern
        log[args.prompt_type][str(args.template_index)]["s_inhibition_head_pattern"] = s_inhibition_head_pattern
        log[args.prompt_type][str(args.template_index)]["name_mover_head_pattern"] = name_mover_head_pattern


    else:
        previous_token_pattern = log[args.prompt_type][str(args.template_index)]["previous_token_pattern"].to(device) 
        duplicate_token_pattern = log[args.prompt_type][str(args.template_index)]["duplicate_token_pattern"].to(device) 
        induction_head_pattern = log[args.prompt_type][str(args.template_index)]["induction_head_pattern"].to(device) 
        s_inhibition_head_pattern = log[args.prompt_type][str(args.template_index)]["s_inhibition_head_pattern"].to(device) 
        name_mover_head_pattern = log[args.prompt_type][str(args.template_index)]["name_mover_head_pattern"].to(device) 

        sentence = log[args.prompt_type][str(args.template_index)]["sentence"] 
        template_O_position = log[args.prompt_type][str(args.template_index)]["template_O_position"] 
        ioi_label = log[args.prompt_type][str(args.template_index)]["ioi_label"] 
        ioi_inputs = log[args.prompt_type][str(args.template_index)]["tokens"].to(device) 
        attention_weights = log[args.prompt_type][str(args.template_index)]["attention_weights"] 

        seq_len = ioi_inputs.shape[1] + 1


    #############################################################################################
    # Loads results log or creates it
    #############################################################################################      
        
    metrics_file = os.path.join(args.results_folder, "heads", f'{ablation_config["name"]}.json')

    if os.path.isfile(metrics_file):
        metrics = json.load(open(metrics_file, "r"))
    else:
        metrics = {}

    if args.prompt_type not in metrics :
        metrics[args.prompt_type] = {}
    
    if str(args.template_index) not in metrics[args.prompt_type]:

        metrics[args.prompt_type][str(args.template_index)] = {}



    #############################################################################################
    # Computes the metrics for each head
    #############################################################################################

    heads_ablation_config = ablation_config["attention_heads"]

    patterns = [
        previous_token_pattern, 
        duplicate_token_pattern, 
        induction_head_pattern, 
        s_inhibition_head_pattern,
        name_mover_head_pattern
    ]
    
    pattern_names = [
        "previous_token_head", 
        "duplicate_token_head", 
        "induction_head", 
        "s_inhibition_head",
        "name_mover_head"
    ]


    tokens = tokenizer.convert_ids_to_tokens(ioi_inputs.squeeze(0))
    tokens = [token.replace("Ġ", "") for token in tokens]

    for pattern, pattern_name in zip(patterns, pattern_names):
        if not os.path.isfile(os.path.join(args.plots_folder, f"{ablation_config['name']}_{pattern_name}")):
            plot_pattern(args, ablation_config["name"], pattern, pattern_name, tokens)

    heads_metrics = metrics[args.prompt_type][str(args.template_index)]

    for layer_idx, heads_indices in heads_ablation_config.items():

        layer_weights = attention_weights[str(layer_idx)].to(device)

        for head_idx in heads_indices:

            head_key = f"{layer_idx}-{head_idx}"

            head_weights = layer_weights[:, head_idx, :, :]

            if not os.path.isfile(os.path.join(args.plots_folder, f"{ablation_config['name']}_{head_key}")):
                plot_pattern(args, ablation_config["name"], head_weights, head_key, tokens)

            if not head_key in heads_metrics:
                heads_metrics[head_key] = {}
                    
                best_pattern_name = None
                best_score = 0.

                for pattern, pattern_name in zip(patterns, pattern_names):

                    #mul = compute_head_attention_similarity_score(head_weights, pattern, error_measure = "mul")
                    abs = compute_head_attention_similarity_score(head_weights, pattern, error_measure = "abs", exclude_bos=False, exclude_current_token=False)

                    abs = round(abs, 3)
                    heads_metrics[head_key][pattern_name] = abs

                    if abs > best_score:
                        best_score = abs
                        best_pattern_name = pattern_name

                heads_metrics[head_key]["head_type"] = best_pattern_name

                print(f"Head ({head_key}) | {best_pattern_name} : {best_score}")

            else:
                print(f"Head ({head_key}) | {heads_metrics[head_key]['head_type']} : {heads_metrics[head_key][heads_metrics[head_key]['head_type']]}")

    metrics[args.prompt_type][str(args.template_index)] = heads_metrics

    json.dump(metrics, open(metrics_file, "w"), indent = 4)


    torch.save(log, detection_path)


def plot_pattern(args, circuit_name, weights, name, tokens):

    plt.figure(figsize = (7, 7))

    im = plt.imshow(weights.squeeze(0).cpu(), cmap="coolwarm", aspect="auto")
    plt.yticks(
        range(len(tokens)), tokens
    )
    plt.xticks(
        range(len(tokens)), tokens, rotation=90
    )
    plt.title("Attention weights pattern")

    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.savefig(os.path.join(args.plots_folder, f"{circuit_name}_{name}"))

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--detection_folder", type = str, default = "detection", help = "Detection folder")
    parser.add_argument("--plots_folder", type = str, default = "plots", help = "Plots folder")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("--results_folder", type = str, default = "results", help = "Results folder")
    parser.add_argument("--configs_folder", type = str, default = "configs", help = "Configurations folder")
    parser.add_argument("-c", "--config", type = str, default = "ioi_paper_ablation.json", help = "Ablation config file")
    parser.add_argument("-i", "--template_index", type = int, default = 0, help = "Index of the template to use (if only one template is selected).")
    parser.add_argument("-p", "--prompt_type", type = str, default = 'BABA', help = "Template to use.")
    parser.add_argument("-s", "--size", type = int, default = 8192, help = "Size of the IOI dataset (power of 2 is simpler for alignment with batch sizes)")
    parser.add_argument("-t", "--num_templates", type = int, default = 1, help = "Number of different templates to use.")
    parser.add_argument("-b", "--batch_size", type = int, default = 512, help = "Size of the batch (can be as large as vram allows as this is eval mode)")

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)
    os.makedirs(args.configs_folder, exist_ok=True)
    os.makedirs(args.detection_folder, exist_ok=True)
    os.makedirs(args.plots_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "heads"), exist_ok=True)

    detect_heads(args)


