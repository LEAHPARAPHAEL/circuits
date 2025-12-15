import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional

def plot_minimality_results(
    results_file: str,
    categories: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    no_abs_value: bool = False,
    figsize: Tuple[int, int] = (15, 6),
    order: Optional[List[Tuple[int, int]]] = None,
    template_key_list = [0],
    save_path: Optional[str] = None
):
    """
    Plot minimality results from a JSON file.
    
    Parameters:
    -----------
    results_file : str
        Path to the JSON results file
    categories : Optional[Dict[str, List[Tuple[int, int]]]]
        Dictionary mapping category names to lists of head tuples (layer, head)
        Example: {"category1": [(2, 9), (1, 3)], "category2": [(5, 5)]}
    no_abs_value : bool
        If True, plot (Logit difference without head - Logit difference with head)
        If False, plot the absolute "Difference in logit difference" from file
    figsize : Tuple[int, int]
        Figure size (width, height)
    order : Optional[List[Tuple[int, int]]]
        List of head tuples specifying the order in which heads should be plotted
        Example: [(9, 9), (10, 0), (5, 5)]
        If provided, only heads in this list will be plotted in the specified order
    """
    
    # Load the results for every template key

    # You need to specify an order
    head_strs = [str(head) for head in order]
    heads = head_strs
    sums = [0]*len(head_strs)

    for template_key in template_key_list:

        results_file_template = results_file.replace("T=0",f"T={template_key}")
        print(f"Trying to load results from: {results_file_template}")
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        for i,head_str in enumerate(head_strs):
            if head_str not in results:
                print(f"Warning: Head {head_str} not found in results file, skipping. THIS WILL GIVE FALSE RESULTS")
                continue
                
            metrics = results[head_str]
            
            if no_abs_value:
                # Compute: Logit difference without head - Logit difference with head
                sums[i] += metrics["Logit difference with head"] - metrics["Logit difference without head"]
            else:
                # Use the absolute difference from the file
                sums[i] += metrics["Difference in logit difference"]

    values = [s / len(template_key_list) for s in sums]

    # Assign colors based on categories
    colors = []
    category_colors = {}
    default_color = 'steelblue'
    
    default_color_present = False
    if categories:
        # Generate distinct colors for each category
        color_palette = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        
        # Create a mapping from head tuple to category
        head_to_category = {}
        for idx, (category_name, head_list) in enumerate(categories.items()):
            category_colors[category_name] = color_palette[idx]
            for head_tuple in head_list:
                head_to_category[head_tuple] = category_name
        
        # Assign colors to each head
        for head_str in heads:
            # Parse head string like "(9, 9)" to tuple (9, 9)
            head_tuple = eval(head_str)
            
            if head_tuple in head_to_category:
                category = head_to_category[head_tuple]
                colors.append(category_colors[category])
            else:
                colors.append(default_color)
                default_color_present = True
    else:
        colors = [default_color] * len(heads)
        default_color_present = True
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x_positions = np.arange(len(heads))
    bars = ax.bar(x_positions, values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('Attention Head (Layer, Head)', fontsize=12, fontweight='bold')
    
    if no_abs_value:
        ax.set_ylabel('Logit Diff With - Logit Diff Without', fontsize=12, fontweight='bold')
        ax.set_title('Impact of Removing Individual Heads (Signed Difference)', fontsize=14, fontweight='bold')
    else:
        ax.set_ylabel('Difference in Logit Difference', fontsize=12, fontweight='bold')
        ax.set_title('Impact of Removing Individual Heads (Absolute Difference)', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(heads, rotation=45, ha='right')
    
    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add legend if categories are provided
    if categories:
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=category_colors[cat], 
                                        edgecolor='black', label=cat) 
                          for cat in categories.keys()]
        if default_color_present:
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=default_color, 
                                                edgecolor='black', label='Uncategorized'))
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return fig, ax
