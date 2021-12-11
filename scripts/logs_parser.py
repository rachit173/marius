#!/bin/python3
import re
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import sys
import json

log_file="./logs.txt"
perf_metrics_label="[Performance Metrics]"


if len(sys.argv) < 3:
    print("Cannot plot. Need range of timestamp as arguments")
    exit(0)

def parse_logs_to_json(logs):
    parsed_logs = []
    init_time = 0

    for rowIdx, log in enumerate(logs):
        cleaned_log=re.split(r'\[|\]', log)
        cleaned_log=[col.strip() for col in cleaned_log if len(col.strip())>0]

        # Get time stamp
        time = datetime.timestamp(datetime.strptime(cleaned_log[1], "%m/%d/%y %H:%M:%S.%f"))
        
        if rowIdx == 0:
            init_time = time
        time_diff = round(time - init_time, 3)
        
        # Convert msg to JSON
        raw_json_msg = cleaned_log[5].replace("(","{").replace(")","}").replace("'","\"")
        
        msg_json = json.loads(raw_json_msg)
        msg_json['global_timestamp'] = time_diff
        
        parsed_logs.append(msg_json)
    return parsed_logs

def plot_timeline(timestamp, messages, minV, maxV, file_name):
    timestamp = timestamp[minV:maxV]
    messages = messages[minV:maxV]

    # Convert date strings (e.g. 2014-10-18) to datetime
    timestamp = [datetime.timestamp(datetime.strptime(d, "%m/%d/%y %H:%M:%S.%f")) for d in timestamp]

    level_range = range(-21, 21, 2)

    levels = np.tile(level_range, int(np.ceil(len(timestamp)/6)))[:len(timestamp)]

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
    logs = [line for line in open(log_file) if perf_metrics_label in line]
    parsed_logs = parse_logs_to_json(logs)

    min_val = int(sys.argv[1])
    max_val = int(sys.argv[2])

    print(parsed_logs[min_val:max_val])
    # file_name = "plot_{}_{}.png".format(minV, maxV)
    # print("Saving plot to " + file_name)
    # plot_timeline(timestamp, messages, minV, maxV, file_name)


parse_and_plot(log_file)