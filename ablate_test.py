from model import AblatableGPT2Model
import argparse
import torch
from ioi_dataset import IOIDataset
import os
from torch.utils.data import TensorDataset, DataLoader
from glob import glob



def compute_abc_means(args):

    size = args.size
    device = args.device

    #############################################################################################
    # Builds the ABC samples
    #############################################################################################

    ioi_samples = IOIDataset(
        prompt_type="BABA",
        N=size,
        nb_templates=args.num_templates,
        seed = 0,
    )

    abc_samples = (
        ioi_samples.gen_flipped_prompts(("IO", "RAND"), seed=1)
        .gen_flipped_prompts(("S", "RAND"), seed=2)
        .gen_flipped_prompts(("S1", "RAND"), seed=3)
    )


    #############################################################################################
    # Finds potential checkpoints
    #############################################################################################

    potential_filepaths = glob(f"{args.abc_means_filepath}_*", root_dir = args.checkpoints_folder)

    if potential_filepaths:
        abc_means_filepath = os.path.join(args.checkpoints_folder, potential_filepaths[0])
        abc_means = torch.load(abc_means_filepath, map_location = device)

        print(f"Successfully loaded previous ABC means from {abc_means_filepath}")

        last_sample_index = abc_means["last_sample_index"]

    else:
        abc_means_filepath = os.path.join(args.checkpoints_folder, f"{args.abc_means_filepath}_0.pth")
        abc_means = None
        last_sample_index = 0


    #############################################################################################
    # Builds the dataset and dataloader
    #############################################################################################

    seq_len = abc_samples.toks.shape[1]

    abc_tokens = abc_samples.toks.long()[last_sample_index:size, :seq_len - 1].to(device)

    abc_dataset = TensorDataset(abc_tokens)

    loader = DataLoader(abc_dataset, batch_size = args.batch_size, shuffle = False)


    #############################################################################################
    # Builds the model
    #############################################################################################^

    model = AblatableGPT2Model.from_pretrained("gpt2")
    model.to(device)
    model.eval()


    #############################################################################################
    # Evaluation loop
    #############################################################################################

    model.accumulate()
    print(f"Evaluating the model on the ABC dataset from index {last_sample_index} to {size}")
    with torch.no_grad():

        current_sample_index = last_sample_index

        for batch in loader:

            inputs = batch[0].to(device)
            
            model(inputs)

            current_sample_index += inputs.size(0)

            if current_sample_index % args.checkpoint_steps == 0:
                abc_means, abc_means_filepath = model.update_accumulation_data(current_sample_index, abc_means_filepath, abc_means)


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the means of the gpt2 model over the ABC dataset")
    parser.add_argument("--checkpoints_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("-r", "--results_folder", type = str, default = "checkpoints", help = "Checkpoints folder")
    parser.add_argument("-a", "--abc_means_filepath", type = str, default = "abc_means", help = "Path to the saved means over the ABC dataset")
    parser.add_argument("-s", "--size", type = int, default = 8192, help = "Size of the ABC dataset (power of 2 is simpler for alignment with batch sizes)")
    parser.add_argument("-c", "--checkpoint_steps", type = int, default = 1024, help = "Number of samples to evaluate between each checkpoint")
    parser.add_argument("-b", "--batch_size", type = int, default = 512, help = "Size of the batch (can be as large as vram allows as this is eval mode)")
    parser.add_argument("-t", "--num_templates", type = int, default = 15, help = "Number of different templates to use.")



    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.checkpoints_folder, exist_ok=True)

    compute_abc_means(args)


