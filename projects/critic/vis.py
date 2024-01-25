import matplotlib.pyplot as plt
import json
import numpy as np


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


def annotate(pps):
        for p in pps:
            height = p.get_height()
            plt.annotate('{}'.format(height),
                         xy=(p.get_x() + p.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')


def baselines():
    plt.clf()

    # GSM8K Plot
    categories = ['Single Step', 'Chain of Thought', 'Deliberated']
    gpt_35 = [0.83, 0.8, 0.57]
    gpt_4 = [0.86, 0.93, 0.71]

    # Adjust the width of the bars as needed
    bar_width = 0.35

    # Calculate the positions for the bars
    x = np.arange(len(categories))

    # Create the grouped bar plot
    pps = plt.bar(x - bar_width/2, gpt_35, bar_width, color="green", label='gpt-3.5', edgecolor="black")
    annotate(pps)
    pps = plt.bar(x + bar_width/2, gpt_4, bar_width, color="#8022b3", label='gpt-4', edgecolor="black")
    annotate(pps)

    # Add labels and a title
    plt.xlabel('Prompts')
    plt.ylabel('Verification Accuracy')
    plt.ylim(0, 1.1)
    plt.title('GSM8K Verification Accuracy')
    plt.xticks(x, categories)
    plt.legend()

    # Display the plot
    plt.savefig("figs/gsm8k_baselines.png")

    ## Now do precision in recall
    """ I'm varying four parameters:
    1. model
    2. prompt
    3. dataset
    4. metric (precision, recall, f1)

    Let's split up datasets into different figures. now just
    1. model
    2. prompt
    3. metric
    """

    # Math baselinse
    plt.clf()

    # MATH Plot
    categories = ['Single Step', 'Chain of Thought', 'Deliberated']
    gpt_35 = [0.56, 0.53, 0.58]
    gpt_4 = [0.69, 0.88, 0.76]

    # Adjust the width of the bars as needed
    bar_width = 0.35

    # Calculate the positions for the bars
    x = np.arange(len(categories))

    # Create the grouped bar plot
    pps = plt.bar(x - bar_width/2, gpt_35, bar_width, color="green", label='gpt-3.5', edgecolor="black")
    annotate(pps)
    pps = plt.bar(x + bar_width/2, gpt_4, bar_width, color="#8022b3", label='gpt-4', edgecolor="black")
    annotate(pps)

    # Add labels and a title
    plt.xlabel('Prompts')
    plt.ylabel('Verification Accuracy')
    plt.ylim(0, 1.1)
    plt.title('MATH Verification Accuracy')
    plt.xticks(x, categories)
    plt.legend()

    # Display the plot
    plt.savefig("figs/math_baselines.png")


def step_level_plots():
    plt.clf()

    colors = ['#5191d2', '#e96f35', '#3ec133',]

    # Sample data for the pie chart
    labels = ["Correct first error pred.", 
              "Early first error pred.", 
              "Late first error pred."]
    sizes = [0.52, 0.06, 0.41]  # The sizes of each slice (percentages)

    # Create a pie chart
    explode = (0.1, 0, 0)
    plt.pie(sizes, labels=None, explode=explode, colors=colors, autopct='%1.1f%%', startangle=90, \
            shadow=True, wedgeprops={'edgecolor': 'grey', 'linewidth': 1.5})
    #plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    # Add a title
    plt.title('gpt-3.5 first error predictions')
    plt.legend(labels, loc="upper left")

    # Show the chart
    plt.savefig("figs/gpt_3.5_first_step_error.png")


    plt.clf()

    # Sample data for the pie chart
    labels = ["Correct first error pred.", 
              "Early first error pred.", 
              "Late first error pred."]
    sizes = [0.43, 0.34, 0.22]  # The sizes of each slice (percentages)

    # Create a pie chart
    plt.pie(sizes, labels=None, explode=explode, colors=colors, autopct='%1.1f%%', startangle=90, \
            shadow=True, wedgeprops={'edgecolor': 'gray', 'linewidth': 1.5})
    #plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    # Add a title
    plt.title('gpt-4 first error predictions')
    plt.legend(labels, loc="upper left")

    # Show the chart
    plt.savefig("figs/gpt_4_first_step_error.png")


def incorrect_step_level_plots():
    plt.clf()

    colors = ['#5191d2', '#e96f35', '#3ec133',]
    explode = (0.1, 0, 0)

    # Sample data for the pie chart
    labels = ["Correct first error pred.", 
              "Early first error pred.", 
              "Late first error pred."]
    sizes = [0.01, 0.01, 0.98]  # The sizes of each slice (percentages)
    

    # Create a pie chart
    plt.pie(sizes, labels=None, explode=explode, colors=colors, autopct='%1.1f%%', startangle=90, \
            shadow=True, wedgeprops={'edgecolor': 'grey', 'linewidth': 1.5})
    #plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    gpt_3_5_offset = 8.8
    plt.annotate(r"$\mathbf{Avg. Offset}$: " + str(gpt_3_5_offset),
             fontsize=11,
             xy=(0.95, 0.9),        # The point you want to annotate
             xytext=(0.95, 0.9),   # The position of the text
             textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='#e2d734'),  # Customize the box
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='blue'))  # Add an arrow

    # Add a title
    plt.title('gpt-3.5 first error predictions on incorrect solutions')
    plt.legend(labels, loc="upper left")

    # Show the chart
    plt.savefig("figs/gpt_3.5_incorrect_first_step_error.png")


    plt.clf()

    # Sample data for the pie chart
    labels = ["Correct first error pred.", 
              "Early first error pred.", 
              "Late first error pred."]
    sizes = [0.24, 0.22, 0.54]  # The sizes of each slice (percentages)

    # Create a pie chart
    plt.pie(sizes, labels=None, explode=explode, colors=colors, autopct='%1.1f%%', startangle=90, \
            shadow=True, wedgeprops={'edgecolor': 'gray', 'linewidth': 1.5})
    #plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    gpt_3_5_offset = 2.4
    plt.annotate(r"$\mathbf{Avg. Offset}$: " + str(gpt_3_5_offset),
             fontsize=11,
             xy=(0.95, 0.9),        # The point you want to annotate
             xytext=(0.95, 0.9),   # The position of the text
             textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='#e2d734'),  # Customize the box
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='blue'))  # Add an arrow

    # Add a title
    plt.title('gpt-4 first error predictions on incorrect solutions')
    plt.legend(labels, loc="upper left")

    # Show the chart
    plt.savefig("figs/gpt_4_incorrect_first_step_error.png")

    # Recheck
    plt.clf()

    # Sample data for the pie chart
    labels = ["Correct first error pred.", 
              "Early first error pred.", 
              "Late first error pred."]
    sizes = [0.28, 0.24, 0.48]  # The sizes of each slice (percentages)

    # Create a pie chart
    plt.pie(sizes, labels=None, explode=explode, colors=colors, autopct='%1.1f%%', startangle=90, \
            shadow=True, wedgeprops={'edgecolor': 'gray', 'linewidth': 1.5})
    #plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    gpt_3_5_offset = 1.7
    plt.annotate(r"$\mathbf{Avg. Offset}$: " + str(gpt_3_5_offset),
             fontsize=11,
             xy=(0.95, 0.9),        # The point you want to annotate
             xytext=(0.95, 0.9),   # The position of the text
             textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='#e2d734'),  # Customize the box
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='blue'))  # Add an arrow

    # Add a title
    plt.title('gpt-4 rechecked first error predictions on incorrect solutions')
    plt.legend(labels, loc="upper left")

    # Show the chart
    plt.savefig("figs/gpt_4_recheck_incorrect_first_step_error.png")


def multi_sample_plots():
    plt.clf()

    # GSM8K Plot
    categories = ['Chain of Thought', 'Multi-sample',]
    gpt_35 = [0.8, 0.88]
    gpt_4 = [0.93, 0.96]

    # Adjust the width of the bars as needed
    bar_width = 0.35

    # Calculate the positions for the bars
    x = np.arange(len(categories))

    # Create the grouped bar plot
    pps = plt.bar(x - bar_width/2, gpt_35, bar_width, color="green", label='gpt-3.5', edgecolor="black")
    annotate(pps)
    pps = plt.bar(x + bar_width/2, gpt_4, bar_width, color="#8022b3", label='gpt-4', edgecolor="black")
    annotate(pps)
    

    # Add labels and a title
    plt.xlabel('Prompts')
    plt.ylabel('Verification Accuracy')
    plt.ylim(0, 1.1)
    plt.title('GSM8K Multi-sample Verification Accuracy')
    plt.xticks(x, categories)
    plt.legend()

    # Display the plot
    plt.savefig("figs/gsm8k_multi_sample.png")

    # Math baselinse
    plt.clf()

    # MATH Plot
    categories = ['Chain of Thought', 'Multi-sample',]
    gpt_35 = [0.55, 0.60]
    gpt_4 = [0.88, 0.82]

    # Adjust the width of the bars as needed
    bar_width = 0.35

    # Calculate the positions for the bars
    x = np.arange(len(categories))

    # Create the grouped bar plot
    pps = plt.bar(x - bar_width/2, gpt_35, bar_width, color="green", label='gpt-3.5', edgecolor="black")
    annotate(pps)
    pps = plt.bar(x + bar_width/2, gpt_4, bar_width, color="#8022b3", label='gpt-4', edgecolor="black")
    annotate(pps)

    # Add labels and a title
    plt.xlabel('Prompts')
    plt.ylabel('Verification Accuracy')
    plt.ylim(0, 1.1)
    plt.title('MATH Multi-sample Verification Accuracy')
    plt.xticks(x, categories)
    plt.legend()

    # Display the plot
    plt.savefig("figs/math_multi_sample.png")


if __name__ == "__main__":
    #step_level_plots()
    #incorrect_step_level_plots()
    baselines()
    multi_sample_plots()