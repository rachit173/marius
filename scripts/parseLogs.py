#!/bin/python3
import re
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import sys

log_file="./logs/marius_worker_0_info.log"

if len(sys.argv) < 3:
    print("Cannot plot")
    exit(0)

def parseLogs(perf_logs):
    parsed_logs = []
    for log in perf_logs:
        res=re.split(r'\[|\]', log)
        res=[col.strip() for col in res if len(col.strip())>0]

        res2 = []
        for idx, col in enumerate(res):
            if idx == 2:
                tmp = [c.split(":")[1] for c in col.split(" ")]
                res2.extend(tmp)
            else:
                res2.append(col)
        
        parsed_logs.append(res2)
    return parsed_logs

def plot_timeline(timestamp, messages, minV, maxV, file_name):
    timestamp = timestamp[minV:maxV]
    messages = messages[minV:maxV]

    # Convert date strings (e.g. 2014-10-18) to datetime
    timestamp = [datetime.timestamp(datetime.strptime(d, "%m/%d/%y %H:%M:%S.%f")) for d in timestamp]

    level_range = range(-21, 21, 2)

    levels = np.tile(level_range,
                    int(np.ceil(len(timestamp)/6)))[:len(timestamp)]

    # Create figure and plot a stem plot with the date
    fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
    ax.set(title="Marius Scaling Timeline")

    ax.vlines(timestamp, 0, levels, color="tab:red")  # The vertical stems.
    ax.plot(timestamp, np.zeros_like(timestamp), "-o",
            color="k", markerfacecolor="w")  # Baseline and markers on it.

    # annotate lines
    for d, l, r in zip(timestamp, levels, messages):
        ax.annotate(r, xy=(d, l),
                    xytext=(-3, np.sign(l)*3), textcoords="offset points",
                    horizontalalignment="right",
                    verticalalignment="bottom" if l > 0 else "top")

    # format xaxis with 4 month intervals
    # ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # remove y axis and spines
    ax.yaxis.set_visible(False)
    # ax.spines[["left", "top", "right"]].set_visible(False)

    ax.margins(y=0.1)
    # plt.show()

    plt.savefig(file_name)


def parse_and_plot(log_file):
    perf_metrics_label="[Performance Metrics]"
    perf_logs = [line for line in open(log_file) if perf_metrics_label in line]
    parsed_logs = parseLogs(perf_logs)

    df = pd.DataFrame(parsed_logs,
    columns =['level', 'timestamp', 'pid', 'tid', 'call_stack', 'label', 'message'])

    # print(df.loc[0:10,:])

    messages = df["message"].tolist()
    timestamp = df["timestamp"].tolist()


    minV = int(sys.argv[1])
    maxV = int(sys.argv[2])

    file_name = "plot_{}_{}.png".format(minV, maxV)
    print("Saving plot to " + file_name)
    plot_timeline(timestamp, messages, minV, maxV, file_name)



parse_and_plot(log_file)