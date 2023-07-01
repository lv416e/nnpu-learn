from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_history(
        results: Tuple[pd.DataFrame, pd.DataFrame] = None,
        colors: Tuple[str, str, str] = ("#F02D3A", "#8AC926", "#1789FC")) -> None:
    """
    TODO: Add comments.
    :param results:
    :param colors:
    :return:
    """
    sns.set(style="darkgrid")
    plt.figure(figsize=(14, 7))
    for idx, result in enumerate(results):
        columns = result.columns.values
        if len(results) >= 2:
            plt.subplot(1, len(results), idx + 1)
        for col in range(3):
            plt.plot(result[columns[0]].values, result[columns[col + 1]].values, label=columns[col + 1], c=colors[col])
            plt.xlabel("Iterations")
            plt.ylabel("loss / risk / acc")
            plt.legend()
    plt.suptitle("History")
    plt.tight_layout()
    plt.show()
