import os

import networkx as nx
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from moviepy import ImageSequenceClip

from conf import RESULTS_DIR, LABELS
import time
import utils

# The iterations whose VLs have to be stored
tt = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

DIR_SUFFIX = "/network-graphs"


class NetworkPlotter:

    def __init__(self, env, LPNet, results_dir):
        self.env = env
        self.graph = LPNet
        self.dir = results_dir
        self.pos = nx.spring_layout(self.graph, k=0.25)

    def Run(self):
        while True:
            print(f"DRAW COLOR CHANGING NETWORK: {self.env.now}")
            if self.env.now <= 5:
                self.draw_color_changing_network()
            yield self.env.timeout(1)

    def draw_color_changing_network(self):
        # Draw the graph with current node states
        colors = {-1: 'blue', 1: 'red'}  # Mapping of states to colors
        node_colors = [colors[self.graph.nodes()[node]["state"]] for node in self.graph.nodes()]
        plt.figure()
        nx.draw(self.graph, self.pos, node_color=node_colors, width=0.5, alpha=0.7, node_size=50)
        plt.title(f"Network graph at time {self.env.now}")
        adapters_count = 0
        non_adapters_count = 0
        for n in self.graph.nodes():
            if self.graph.nodes()[n]["state"] == +1:
                adapters_count += 1
            else:
                non_adapters_count += 1
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label=f'Number of adapters: {adapters_count}'),
            Patch(facecolor='blue', edgecolor='black', label=f'Number of non-adapters: {non_adapters_count}')
        ]
        # Add legend to the plot
        plt.legend(handles=legend_elements, loc='lower center')
        utils.create_if_not_exist(f"{RESULTS_DIR}/network-graphs")
        plt.savefig(f"{RESULTS_DIR}/network-graphs/graph_{self.env.now}.png", dpi=1000)

    def create_animation(self):
        complete_path = self.dir + DIR_SUFFIX
        png_files = sorted([f for f in os.listdir(complete_path) if f.endswith('.png')])
        png_paths = [os.path.join(complete_path, f) for f in png_files]
        #images = [Image.open(png) for png in png_paths]

        clip = ImageSequenceClip(png_paths, fps=2)  # Set frames per second (adjust to your needs)

        # Save as MP4
        clip.write_videofile(complete_path + '/network_animation.mp4', codec="libx264")
        # Save as GIF
        # images[0].save(
        #     complete_path + '/network_animation.gif',
        #     save_all=True,
        #     append_images=images[1:],
        #     duration=500,  # Duration per frame in milliseconds
        #     loop=0
        # )
