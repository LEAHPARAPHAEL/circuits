import json


def compile_json_to_latex_table(file_name):
    data = json.load(open(file_name))
    output = "Category & Ablated model f & Ablated circuit f & Incompleteness score \\\\\n"
    for key in data.keys():
        if key in ["name","ablation_set_type","completeness_config"]:
            continue
        else:
            output += f"{key}"
            for metric in data[key].keys():
                output += f"& {data[key][metric]:.2f} "
            output += "\\\\"
    return output


print(compile_json_to_latex_table("results/completeness/completeness_BABA_T=0_type=set_adversarial.json"))
