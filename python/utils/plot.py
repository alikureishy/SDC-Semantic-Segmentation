#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os.path
import shutil

def plot_loss(runs_dir, losses, plot_file_name):
    _, axes = plt.subplots()
    plt.plot(range(0, len(losses)), losses)
    plt.title('Cross-entropy loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
    os.makedirs(runs_dir)

    output_file = os.path.join(runs_dir, plot_file_name + ".png")
    plt.savefig(output_file)
