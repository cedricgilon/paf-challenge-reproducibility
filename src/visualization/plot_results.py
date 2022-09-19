from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ALPHA = 0.3


def create_figures(exp_metrics, csv_file, filename, figures_folder, metrics):
    df = pd.read_csv(csv_file)

    create_scatter_plot(df, exp_metrics, filename, figures_folder)

    create_box_plot(df, exp_metrics["accuracy"], "accuracy", filename, figures_folder)
    create_box_plot(df, exp_metrics["sensitivity"], "sensitivity", filename, figures_folder)
    create_box_plot(df, exp_metrics["specificity"], "specificity", filename, figures_folder)


def create_scatter_plot(df, exp_metrics, model_name, figures_folder):
    plt.scatter(exp_metrics["specificity"], exp_metrics["sensitivity"], marker="*", s=100, alpha=1,
                color="red", label="reported values")

    markers = ["o", "+", "D", "^", "s", "<"]
    colors = ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7", "#000000"]
    for experience, color, marker in zip(df.experience_name.unique(), colors, markers):
        df_exp = df[df.experience_name == experience]
        plt.scatter(df_exp.specificity.values, df_exp.sensitivity.values,
                    label=experience, alpha=ALPHA, marker=marker, color=color)

    plt.rcParams["figure.figsize"] = (15, 15)
    plt.xlabel('specificity')
    plt.ylabel('sensitivity')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    # plt.show()
    plt.savefig(Path(figures_folder, f"{model_name}_scatter.png"), dpi=300)
    plt.clf()


def create_box_plot(df, expected_metric, metric_name, model_name, figures_folder):
    plt.rcParams["figure.figsize"] = (10, 10)
    df_group = df.groupby('experience_name', sort=False).agg({metric_name: lambda x: list(x)})
    metric_list = df_group[metric_name].values
    plt.boxplot(metric_list)
    plt.ylim(-0.05, 1.05)
    plt.ylabel(metric_name)

    plt.axhline(y=expected_metric, color='r', linestyle='--', label=f'reported {metric_name}')
    labels = df_group.index.values
    plt.xticks(range(1, len(labels) + 1), labels, rotation=15)

    plt.legend()

    plt.savefig(Path(figures_folder, f"{model_name}_{metric_name}.png"), dpi=300)
    plt.clf()
