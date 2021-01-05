# !!! https://www.redblobgames.com/maps/terrain-from-noise/

# https://jackmckew.dev/3d-terrain-in-python.html
# https://medium.com/@yvanscher/playing-with-perlin-noise-generating-realistic-archipelagos-b59f004d8401
# https://engineeredjoy.com/blog/perlin-noise/
# https://heredragonsabound.blogspot.com/2019/02/perlin-noise-procedural-content.html
# https://www.reddit.com/r/proceduralgeneration/comments/kaen7h/new_video_on_procedural_island_noise_generation/gfjmgen/

import math
import random
import time

import numpy as np
from noise import pnoise2
from numpy.core.fromnumeric import shape
from PIL import Image


def timeit(func, args=[], kwargs={}):
    s = time.time()
    func(*args, **kwargs)
    return time.time() - s


class Color:
    DEEP_OCEAN = (94, 129, 172)
    OCEAN = (115, 146, 183)
    BEACH = (235, 203, 139)
    LAND = (143, 176, 115)
    FOREST = (163, 190, 140)
    MOUNTAIN = (76, 86, 106)
    HIGH_MOUNTAIN = (121, 133, 159)
    SNOW = (216, 222, 233)
    CITY = (255, 0, 0)


class ShapeFunction:
    @staticmethod
    def upper(shape):
        return np.ones(shape)

    @staticmethod
    def lower(shape):
        return np.zeros(shape)


class EuclidianShape(ShapeFunction):
    @staticmethod
    def upper(shape):
        denom = math.sqrt(0.5)

        def func(x, y):
            ny = y / shape[0] - 0.5
            nx = x / shape[1] - 0.5

            return 1 - (math.sqrt((nx * nx) + (ny * ny)) / denom)

        return np.fromfunction(np.vectorize(func), shape)


class SquareBumpShape(ShapeFunction):
    @staticmethod
    def upper(shape):
        def func(x, y):
            ny = y / shape[0] - 0.5
            nx = x / shape[1] - 0.5

            return ((1 - nx * nx) * (1 - ny * ny)) ** 3

        return np.fromfunction(np.vectorize(func), shape)


class Map:
    def __init__(
        self,
        width,
        height,
        frequency=8,  # zoom
        octaves=5,  # levels of stacked noise
        redistribution=1.0,
        persistence=0.5,
        lacunarity=2,
        seed=None,
        shape_func=SquareBumpShape,
    ):
        self.shape = (height, width)
        self.frequency = frequency
        self.octaves = octaves
        self.redistribution = redistribution
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.shape_function = shape_func

        if not seed:
            self.seed = int(random.random() * 32)

        self.elevation = []

        self.coordinate_map = np.indices(self.shape).transpose((1, 2, 0))
        self.coordinate_map = self.coordinate_map / self.shape - 0.5

        print("generation", timeit(self._generate))
        print("shaping", timeit(self._shape))
        print("normalization", timeit(self._normalize))
        print("coloring", timeit(self.to_image, ["out.bmp"]))

    def _generate(self):
        def pixel(i, j):
            nx = i / self.shape[1] - 0.5
            ny = j / self.shape[0] - 0.5

            # noise generation
            return pnoise2(
                self.frequency * nx,
                self.frequency * ny,
                octaves=self.octaves,
                persistence=self.persistence,
                lacunarity=self.lacunarity,
                base=self.seed,
            )

        # generate 2d lists of noise
        self.elevation = [
            [pixel(x, y) for x in range(self.shape[1])] for y in range(self.shape[0])
        ]

        # convert 2d lists to numpy array
        self.elevation = np.array(self.elevation)

        # normalize from [-1, 1] to [0, 1]
        self.elevation = self.elevation / 2.0 + 0.5

        # raising to the power of one will have no effect, so skip this calculation
        if self.redistribution != 1:
            # uses an exponent to adjust the distribution (think height curve)
            self.elevation **= self.redistribution

    def _shape(self):
        """
        upp = self.shape_function.upper(self.shape)
        low = self.shape_function.lower(self.shape)
        """

        upp = self.shape_function.upper(self.shape)
        low = self.shape_function.lower(self.shape)

        self.elevation = low + self.elevation * (upp - low)

    def _normalize(self):
        """Normalize elevation to [0,1]."""
        self.elevation = (self.elevation - np.min(self.elevation)) / np.ptp(
            self.elevation
        )

    def to_image(self, filename, grayscale=False):
        def color(e):
            if e >= 0.85:
                return Color.SNOW
            elif e >= 0.75:
                return Color.HIGH_MOUNTAIN
            elif e >= 0.65:
                return Color.MOUNTAIN
            elif e >= 0.50:
                return Color.FOREST
            elif e >= 0.37:
                return Color.LAND
            elif e >= 0.35:
                return Color.BEACH
            elif e >= 0.20:
                return Color.OCEAN
            else:
                return Color.DEEP_OCEAN

        if grayscale:
            self._normalize()  # TODO: maybe remove?
            grayscale_world = self.elevation * 255.0
            im = Image.fromarray(np.uint8(grayscale_world), "L").save(filename)
            return

        image_data = [
            color(self.elevation[y][x])
            for y in range(self.shape[0])
            for x in range(self.shape[1])
        ]

        im = Image.new("RGB", self.shape[::-1], "white")
        im.putdata(image_data)
        im.save(filename)

    def to_plot(self, show_shape=False):
        import matplotlib.pyplot as plt

        lin_x = np.linspace(0, 1, self.shape[1], endpoint=False)
        lin_y = np.linspace(0, 1, self.shape[0], endpoint=False)
        x, y = np.meshgrid(lin_x, lin_y)

        fig = plt.figure()

        if show_shape:
            ax1 = fig.add_subplot(312, projection="3d")
            ax1.set_zlim([0, 1.2])
            ax1.plot_surface(x, y, self.elevation, cmap="terrain")

            ax2 = fig.add_subplot(311, projection="3d")
            ax2.set_zlim([0, 1.2])
            ax2.plot_surface(
                x,
                y,
                self.shape_function.upper(self.shape),
                cmap="terrain",
            )

            ax3 = fig.add_subplot(313, projection="3d")
            ax3.set_zlim([0, 1.2])
            ax3.plot_surface(
                x,
                y,
                self.shape_function.lower(self.shape),
                cmap="terrain",
            )

            def on_move(event):
                # https://stackoverflow.com/questions/41615448/pyplot-share-axis-between-3d-subplots
                if event.inaxes == ax1:
                    if ax1.button_pressed in ax1._rotate_btn:
                        ax2.view_init(elev=ax1.elev, azim=ax1.azim)
                        ax3.view_init(elev=ax1.elev, azim=ax1.azim)
                    elif ax1.button_pressed in ax1._zoom_btn:
                        ax2.set_xlim3d(ax1.get_xlim3d())
                        ax2.set_ylim3d(ax1.get_ylim3d())
                        ax2.set_zlim3d(ax1.get_zlim3d())
                        ax3.set_xlim3d(ax1.get_xlim3d())
                        ax3.set_ylim3d(ax1.get_ylim3d())
                        ax3.set_zlim3d(ax1.get_zlim3d())
                elif event.inaxes == ax2:
                    if ax2.button_pressed in ax2._rotate_btn:
                        ax1.view_init(elev=ax2.elev, azim=ax2.azim)
                        ax3.view_init(elev=ax2.elev, azim=ax2.azim)
                    elif ax2.button_pressed in ax2._zoom_btn:
                        ax1.set_xlim3d(ax2.get_xlim3d())
                        ax1.set_ylim3d(ax2.get_ylim3d())
                        ax1.set_zlim3d(ax2.get_zlim3d())
                        ax3.set_xlim3d(ax2.get_xlim3d())
                        ax3.set_ylim3d(ax2.get_ylim3d())
                        ax3.set_zlim3d(ax2.get_zlim3d())
                elif event.inaxes == ax3:
                    if ax3.button_pressed in ax3._rotate_btn:
                        ax2.view_init(elev=ax3.elev, azim=ax3.azim)
                        ax1.view_init(elev=ax3.elev, azim=ax3.azim)
                    elif ax3.button_pressed in ax3._zoom_btn:
                        ax1.set_xlim3d(ax3.get_xlim3d())
                        ax1.set_ylim3d(ax3.get_ylim3d())
                        ax1.set_zlim3d(ax3.get_zlim3d())
                        ax2.set_xlim3d(ax3.get_xlim3d())
                        ax2.set_ylim3d(ax3.get_ylim3d())
                        ax2.set_zlim3d(ax3.get_zlim3d())
                else:
                    return
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", on_move)
        else:
            ax1 = fig.add_subplot(111, projection="3d")
            ax1.set_zlim([0, 1.2])
            ax1.plot_surface(x, y, self.elevation, cmap="terrain")

        plt.show()


if __name__ in "__main__":
    m = Map(2048, 2048, shape_func=EuclidianShape, redistribution=1.1)
