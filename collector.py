import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import PolygonSelector, RectangleSelector


class PolygonCollector:
    def __init__(self):
        self.selector = None

    def collect(self, img, title='', max_count=-1):
        rects = []
        def line_select_callback(points):
            rects.append(points)
            plt.close()
            if max_count > 0 and len(rects) >= max_count:
                return
            fig, ax_ = plt.subplots()
            ax_.imshow(img)
            ax_.set_title(title)
            for poly in rects:
                xs, ys = zip(*(poly + [poly[0], ]))  # create lists of x and y values
                plt.fill(xs, ys)
            self.selector.disconnect_events()
            self.selector = PolygonSelector(ax_, line_select_callback)
            plt.show()

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(title)
        self.selector = PolygonSelector(ax, line_select_callback)
        plt.show()
        return [[list(i) for i in points] for points in rects]


class RectangleCollector:
    def __init__(self):
        self.selector = None

    def collect(self, img, title='', max_count=-1):
        rects = []

        def line_select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            rects.append((int(x1), int(y1), int(x2), int(y2)))
            plt.close()
            if max_count > 0 and len(rects) >= max_count:
                return
            fig, ax_ = plt.subplots()
            ax_.imshow(img)
            ax_.set_title(title)
            for x1, y1, x2, y2 in rects:
                rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                ax_.add_patch(rect)
            self.selector.disconnect_events()
            self.selector = RectangleSelector(ax_, line_select_callback)
            plt.show()

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(title)
        self.selector = RectangleSelector(ax, line_select_callback)
        plt.show()
        return rects
