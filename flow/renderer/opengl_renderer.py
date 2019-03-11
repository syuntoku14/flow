"""Contains the pyglet renderer class."""

import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import cv2
import imutils
import os
from os.path import expanduser
import time
import copy
import warnings
HOME = expanduser("~")

import os

MAX_BUF = 1024
myfifo = "/tmp/fifopipe2"
os.mkfifo(myfifo, 0o666)

def write_message(msg):
    msg = msg.encode('utf-8');
    fd = os.open(myfifo, os.O_WRONLY)
    print("fd ", fd);
    #msg=b"This is the string to be reversed"
    os.write(fd, msg)
    os.close(fd)
    os.unlink(myfifo)


def read_message():
    myfifo = "/tmp/fifopipe"
    buf = ""
    flag = False
    while not flag:
        try:
            fd = os.open(myfifo, os.O_RDONLY | os.O_NONBLOCK)
            flag = True
        except FileNotFoundError:
            pass

    while len(buf) == 0:
        buf = os.read(fd, MAX_BUF)

    #print("Received:", buf.decode())
    os.close(fd)
    return buf.decode()


def truncate_colormap(cmap, minval=0.25, maxval=0.75, n=100):
    """Truncate a matplotlib colormap.
    """

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


class OpenGLRenderer(object):

    def __init__(self, network, mode,
                 save_render=False,
                 path=HOME+"/flow_rendering",
                 sight_radius=50,
                 show_radius=False,
                 pxpm=2,
                 alpha=1.0):
        """Instantiate a pyglet renderer class.

            Parameters
            ----------
            network: list
                A list of road network polygons
            mode: str or bool
                False: no rendering
                True: delegate rendering to sumo-gui for back-compatibility
                "gray": static grayscale rendering, which is good for training
                "dgray": dynamic grayscale rendering
                "rgb": static RGB rendering
                "drgb": dynamic RGB rendering, which is good for visualization
            save_render: bool
                Specify whether to save rendering data to disk
            path: str
                Specify where to store the rendering data
            sight_radius: int
                Set the radius of observation for RL vehicles (meter)
            show_radius: bool
                Specify whether to render the radius of RL observation
            pxpm: int
                Specify rendering resolution (pixel / meter)
            alpha: int
                Specify opacity of the alpha channel.
                1.0 is fully opaque; 0.0 is fully transparent.
        """
        self.mode = mode
        if self.mode not in [True, False, "rgb", "drgb", "gray", "dgray"]:
            raise ValueError("Mode %s is not supported!" % self.mode)
        self.save_render = save_render
        self.path = path + '/' + time.strftime("%Y-%m-%d-%H%M%S")
        if self.save_render:
            if not os.path.exists(path):
                os.mkdir(path)
            os.mkdir(self.path)
            self.data = [network]
        self.sight_radius = sight_radius
        self.pxpm = pxpm  # Pixel per meter
        self.show_radius = show_radius
        self.alpha = alpha
        self.enable_alpha()
        self.time = 0

        self.lane_polys = copy.deepcopy(network)
        lane_polys_flat = [pt for poly in network for pt in poly]

        polys_x = np.asarray(lane_polys_flat[::2])
        width = int(polys_x.max() - polys_x.min())
        shift = polys_x.min() - 2
        scale = (width - 4) / width
        self.width = (width + 2*self.sight_radius) * self.pxpm
        self.x_shift = shift - self.sight_radius
        self.x_scale = scale

        polys_y = np.asarray(lane_polys_flat[1::2])
        height = int(polys_y.max() - polys_y.min())
        shift = polys_y.min() - 2
        scale = (height - 4) / height
        self.height = (height + 2*self.sight_radius) * self.pxpm
        self.y_shift = shift - self.sight_radius
        self.y_scale = scale

        self.lane_colors = []
        for lane_poly in self.lane_polys:
            lane_poly[::2] = [(x-self.x_shift)*self.x_scale*self.pxpm
                              for x in lane_poly[::2]]
            lane_poly[1::2] = [(y-self.y_shift)*self.y_scale*self.pxpm
                               for y in lane_poly[1::2]]
            color = [c for _ in range(int(len(lane_poly)/2))
                     for c in [224, 224, 224, int(self.alpha*255)]]
            self.lane_colors.append(color)

        try:
            self.window = self.create_window(
                width=self.width, height=self.height,
                color=(0.125, 0.125, 0.125, self.alpha)
            )
            self.enable_alpha()
            self.clear_window()
            self.add_lane_polys()
            image_data = buffer.get_image_data()
            frame = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            frame = frame.reshape(buffer.height, buffer.width, 4)
            self.frame = frame[::-1, :, 0:3][..., ::-1]
            self.network = self.frame.copy()
            print("Rendering with Pyglet with frame size",
                  (self.width, self.height))
        except ImportError:
            self.window = None
            self.frame = None
            warnings.warn("Cannot access display. Aborting.", ResourceWarning)

    def render(self,
               human_orientations,
               machine_orientations,
               human_dynamics,
               machine_dynamics,
               human_logs,
               machine_logs,
               save_render=None,
               sight_radius=None,
               show_radius=None,
               sleep=0):
        """Update the rendering frame.

            Parameters
            ----------
            human_orientations: list
                A list contains orientations of all human vehicles
                An orientation is a list contains [x, y, angle].
            machine_orientations: list
                A list contains orientations of all RL vehicles
                An orientation is a list contains [x, y, angle].
            human_dynamics: list
                A list contains the speed of all human vehicles normalized by
                max speed, i.e., speed/max_speed
                This is used to dynamically color human vehicles based on its
                velocity.
            machine_dynamics: list
                A list contains the speed of all RL vehicles normalized by
                max speed, i.e., speed/max_speed
                This is used to dynamically color RL vehicles based on its
                velocity.
            human_logs: list
                A list contains the timestep (ms), timedelta (ms), and id of
                all human vehicles
            machine_logs: list
                A list contains the timestep (ms), timedelta (ms), and id of
                all RL vehicles
            save_render: bool
                Specify whether to Specify whether to save rendering data to
                disk
            sight_radius: int
                Set the radius of observation for RL vehicles (meter)
            show_radius: bool
                Specify whether to render the radius of RL observation
            sleep: float
                Specify the rendering delay in second
        """

        if save_render is None:
            save_render = self.save_render
        if sight_radius is not None:
            sight_radius = sight_radius * self.pxpm
        else:
            sight_radius = self.sight_radius
        if show_radius is None:
            show_radius = self.show_radius

        if save_render:
            _human_orientations = copy.deepcopy(human_orientations)
            _machine_orientations = copy.deepcopy(machine_orientations)
            _human_dynamics = copy.deepcopy(human_dynamics)
            _machine_dynamics = copy.deepcopy(machine_dynamics)
            _human_logs = copy.deepcopy(human_logs)
            _machine_logs = copy.deepcopy(machine_logs)

        self.time += 1
        time.sleep(sleep)

        self.enable_alpha()
        self.clear_window()

        self.add_lane_polys()
        if "drgb" in self.mode:
            human_cmap = truncate_colormap(cm.Greens, 0.2, 0.8)
            machine_cmap = truncate_colormap(cm.Blues, 0.2, 0.8)
            human_conditions = [
                (255*np.array(human_cmap(d)[:3]+(self.alpha,)))\
                .astype(np.uint8).tolist()
                for d in human_dynamics]
            machine_conditions = [
                (255*np.array(machine_cmap(d)[:3]+(self.alpha,)))\
                .astype(np.uint8).tolist()
                for d in machine_dynamics]

        elif "dgray" in self.mode:
            human_cmap = truncate_colormap(cm.binary, 0.55, 0.95)
            machine_cmap = truncate_colormap(cm.binary, 0.05, 0.45)
            human_conditions = [
                (255*np.array(human_cmap(d)[:3]+(self.alpha,)))\
                .astype(np.uint8).tolist()
                for d in human_dynamics]
            machine_conditions = [
                (255*np.array(machine_cmap(d)[:3]+(self.alpha,)))\
                .astype(np.uint8).tolist()
                for d in machine_dynamics]

        elif "rgb" in self.mode:
            human_conditions = [
                [0, 225, 0, int(255*self.alpha)] for d in human_dynamics]
            machine_conditions = [
                [0, 150, 200, int(255*self.alpha)] for d in machine_dynamics]

        elif "gray" in self.mode:
            human_conditions = [
                [100, 100, 100, int(255*self.alpha)] for d in human_dynamics]
            machine_conditions = [
                [150, 150, 150, int(255*self.alpha)] for d in machine_dynamics]

        self.add_vehicle_polys(human_orientations,
                               human_conditions, 0)
        if show_radius:
            self.add_vehicle_polys(machine_orientations,
                                   machine_conditions,
                                   sight_radius)
        else:
            self.add_vehicle_polys(machine_orientations,
                                   machine_conditions, 0)

        # buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        # image_data = buffer.get_image_data()
        # frame = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        # frame = frame.reshape(buffer.height, buffer.width, 4)
        # self.frame = frame[::-1, :, 0:3][..., ::-1]
        # self.window.flip()

        if save_render:
            cv2.imwrite("%s/frame_%06d.png" %
                        (self.path, self.time), self.frame)
            self.data.append([_human_orientations, _machine_orientations,
                              _human_dynamics, _machine_dynamics,
                              _human_logs, _machine_logs])
        if "gray" in self.mode:
            return self.frame[:, :, 0]
        else:
            return self.frame

    def get_sight(self, orientation, id, sight_radius=None, save_render=None):
        """Return the local observation of a vehicle.

            Parameters
            ----------
            orientation: list
                An orientation is a list contains [x, y, angle]
            id: str
                The vehicle to observe for
            sight_radius: int
                Set the radius of observation for RL vehicles (meter)
            save_render: bool
                Specify whether to save rendering data to disk
        """

        if sight_radius is not None:
            sight_radius = sight_radius * self.pxpm
        else:
            sight_radius = self.sight_radius * self.pxpm
        if save_render is None:
            save_render = self.save_render

        x, y, ang = orientation
        x = (x-self.x_shift)*self.x_scale*self.pxpm
        y = (y-self.y_shift)*self.y_scale*self.pxpm
        x_med = x
        y_med = self.height - y
        x_min = int(x_med - sight_radius)
        y_min = int(y_med - sight_radius)
        x_max = int(x_med + sight_radius)
        y_max = int(y_med + sight_radius)
        fixed_sight = self.frame[y_min:y_max, x_min:x_max]
        height, width = fixed_sight.shape[0:2]
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (int(sight_radius), int(sight_radius)),
                   int(sight_radius), (255, 255, 255), thickness=-1)
        rotated_sight = cv2.bitwise_and(fixed_sight, fixed_sight, mask=mask)
        rotated_sight = imutils.rotate(rotated_sight, ang)

        if save_render:
            cv2.imwrite("%s/sight_%s_%06d.png" %
                        (self.path, id, self.time),
                        rotated_sight)
        if "gray" in self.mode:
            return rotated_sight[:, :, 0]
        else:
            return rotated_sight

    def close(self):
        """Terminate the renderer.
        """

        if self.save_render:
            np.save("%s/data_%06d.npy" % (self.path, self.time), self.data)
        self.close_window()

    def add_lane_polys(self):
        """Render road network polygons.
        """
        for lane_poly, lane_color in zip(self.lane_polys, self.lane_colors):
            self.draw_line(shape=lane_poly, color=lane_color)

    def add_vehicle_polys(self, orientations, colors, sight_radius):
        """Render vehicle polygons.

            Parameters
            ----------
            orientation: list
                A list of orientations
                An orientation is a list contains [x, y, angle].
            colors: list
                A list of colors corresponding to the vehicle orientations
            sight_radius: int
                Set the radius of observation for RL vehicles (meter)
        """
        for orientation, color in zip(orientations, colors):
            x, y, ang = orientation
            x = (x-self.x_shift)*self.x_scale*self.pxpm
            y = (y-self.y_shift)*self.y_scale*self.pxpm
            self._add_vehicle_poly_triangle((x, y), ang, 5, color)
            self._add_vehicle_poly_circle((x, y), sight_radius, color)

    def _add_vehicle_poly_triangle(self, center, angle, size, color):
        """Internal pyglet method to render a vehicle as a triangle.

            Parameters
            ----------
            center: tuple
                The center coordinate of the vehicle
            angle: float
                The angle of the vehicle
            size: int
                The size of the rendered triangle
            color: list
                The color of the vehicle  [r, g, b].
        """
        cx, cy = center
        ang = np.radians(angle)
        s = size*self.pxpm
        pt1 = [cx, cy]
        pt1_ = [cx - s*self.x_scale*np.sin(ang),
                cy - s*self.y_scale*np.cos(ang)]
        pt2 = [pt1_[0] + 0.25*s*self.x_scale*np.sin(np.pi/2-ang),
               pt1_[1] - 0.25*s*self.y_scale*np.cos(np.pi/2-ang)]
        pt3 = [pt1_[0] - 0.25*s*self.x_scale*np.sin(np.pi/2-ang),
               pt1_[1] + 0.25*s*self.y_scale*np.cos(np.pi/2-ang)]
        vertex_list = []
        vertex_color = []
        for point in [pt1, pt2, pt3]:
            vertex_list += point
            vertex_color += color
        self.draw_polygon(shape=[pt1, pt2, pt3], color=color)

    def _add_vehicle_poly_circle(self, center, radius, color):
        """Internal pyglet method to render a vehicle as a circle or to render
        its observation radius.

            Parameters
            ----------
            center: tuple
                The center coordinate of the vehicle
            radius: float
                The size of the rendered vehicle or the radius of observation
            color: list
                The color of the vehicle  [r, g, b].
        """
        if radius == 0:
            return
        cx, cy = center
        radius = radius * self.pxpm
        pxpm = int(self.pxpm*50)
        vertex_list = []
        for idx in range(pxpm):
            angle = np.radians(float(idx)/pxpm * 360.0)
            x = radius*self.x_scale*np.cos(angle) + cx
            y = radius*self.y_scale*np.sin(angle) + cy
            vertex_list += [x, y]
        self.draw_line(shape=vertex_list, color=color)

    def enable_alpha(self, enable=True):
        """Enable alpha channel for coloring."""
        return False


    def clear_window(self):
        # handshake
        write_message("1!")  # 1 means clear window
        read_message()

    def create_window(self, width, height, color):
        # handshake
        write_message("2!") # 2 means create window
        read_message()

        # send Parameters
        data = [width, height] + color
        write_message(",".join(data) + "!")
        read_message()

    def draw_line(self, shape, color):
        # handshake
        write_message("3!")
        read_message()

        for i, point in enumerate(shape):
            status = "1" if i == len(shape)-1 else "0"
            write_message(",".join(point) + "," + status + "!")
            read_message()

        write_message(",".join(color) + "!")
        read_message()

    def draw_polygon(self, shape, color):
        # handshake
        write_message("4!")
        read_message()

        for i, point in enumerate(shape):
            status = "1" if i == len(shape)-1 else "0"
            write_message(",".join(point) + "," + status + "!")
            read_message()

        write_message(",".join(color) + "!")
        read_message()