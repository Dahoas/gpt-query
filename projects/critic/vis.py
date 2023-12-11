import matplotlib.pyplot as plt
import json


def plot():
    results = {}
    results["single_step"] = json.load(open("results/gpt_4_turbo_gsm8k_eval_single_step_.json", "r"))
    results["step_by_step"] = json.load(open("results/gpt_4_turbo_gsm8k_eval_step_by_step_.json", "r"))

    plot_results = {}
    for k, v in results.items():
        plot_results[k] = v["accuracy"]

    plt.clf()
    plt.bar(plot_results.keys(), plot_results.values(), width=0.4)
    plt.savefig("figs/eval_accs.png")


if __name__ == "__main__":
    plot()