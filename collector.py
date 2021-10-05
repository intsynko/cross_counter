import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector


class PolygonCollector:
    def __init__(self):
        self.selector = None

    def collect(self, img, title=''):
        rects = []
        def line_select_callback(points):
            rects.append(points)
            plt.close()

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
        return rects

