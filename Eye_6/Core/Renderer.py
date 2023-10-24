"""
2D rendering framework
"""
import os
import sys

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

from gym import error

try:
    import pyglet
except ImportError as e:
    raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError('''
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    ''')

try:
    from pyglet.window import key
except ImportError as e:
    raise ImportError('''
    Error occurred while running `from pyglet.window import key`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    ''')


import math
from math import *
import numpy as np
import quaternion

RAD2DEG = 57.29577951308232


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))


def get_window(width, height, display, **kwargs):
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens() #available screens
    config = screen[0].get_best_config() #selecting the first screen
    context = config.create_context(None) #create GL context

    return PygletWindow(width=width, height=height, display=display, config=config, context=context, **kwargs)


class PygletWindow(pyglet.window.Window):

    def setLock(self,state): self.lock = state; self.set_exclusive_mouse(state)
    lock = False; mouse_lock = property(lambda self: self.lock, setLock)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(300,200)
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        # pyglet.clock.schedule_interval(self.update, 0.01)
        # pyglet.clock.schedule(self.update)

        self.player = Player((-2, 0, 0),(0, -120))

    def on_mouse_motion(self,x,y,dx,dy):
        if self.mouse_lock: self.player.mouse_motion(dx, dy)

    def on_key_press(self, KEY, MOD):
        if KEY == key.ESCAPE: self.close()
        elif KEY == key.E: self.mouse_lock = not self.mouse_lock
        elif KEY == key.I:
            print(self.player.pos, self.player.rot)

    def update(self, dt):
        # print(dt)
        self.player.update(1/120, self.keys)

    def get_player_position(self):
        return self.player.pos

    def get_player_rotation(self):
        return self.player.rot


class Player:
    def __init__(self,pos=(0,0,0),rot=(0,0)):
        self.pos = list(pos)
        self.rot = list(rot)

    def mouse_motion(self,dx,dy):
        dx/=8; dy/=8; self.rot[0]+=dy; self.rot[1]-=dx
        if self.rot[0]>90: self.rot[0] = 90
        elif self.rot[0]<-90: self.rot[0] = -90

    def update(self,dt,keys):
        s = dt*10
        rotY = -self.rot[1]/180*math.pi
        dx,dz = s*math.sin(rotY),s*math.cos(rotY)
        if keys[key.W]: self.pos[0]+=dx; self.pos[2]-=dz
        if keys[key.S]: self.pos[0]-=dx; self.pos[2]+=dz
        if keys[key.A]: self.pos[0]-=dz; self.pos[2]-=dx
        if keys[key.D]: self.pos[0]+=dz; self.pos[2]+=dx
        if keys[key.SPACE]: self.pos[1]+=s
        if keys[key.LSHIFT]: self.pos[1]-=s


class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height
        self.window = get_window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def get_update_func(self): # TODO hate myself because of this
        return self.window.update

    def close(self):
        if self.isopen and sys.meta_path:
            # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
            self.window.close()
            self.isopen = False

    def window_closed_by_user(self):
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    def add_geom(self, geom):
        geom.do_batch()
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        glClearColor(0.5, 0.7, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.set3d()
        self.push(self.window.get_player_position(), self.window.get_player_rotation())
        # self.push((-0.05, 4.25, -2.5),(-54, -135))
        self.transform.enable()

        for geom in self.geoms:
            geom.render()

        self.transform.disable()
        glPopMatrix()

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else self.isopen

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

    def __del__(self):
        self.close()

    #added
    def push(self,pos,rot): glPushMatrix(); glRotatef(-rot[0],1,0,0); glRotatef(-rot[1],0,1,0); glTranslatef(-pos[0],-pos[1],-pos[2],)
    def Projection(self): glMatrixMode(GL_PROJECTION); glLoadIdentity()
    def Model(self): glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    def set2d(self): self.Projection(); gluOrtho2D(0,self.width,0,self.height); self.Model()
    def set3d(self): self.Projection(); gluPerspective(90,self.width/self.height,0.05,1000); self.Model()


class Geom(object):
    def __init__(self):
        # self._color = Color((0, 0, 0, 1.0))
        self.attrs = []

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b):
        self._color.vec4 = (r, g, b, 1)


class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0, 0.0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        self.set_translation(*translation)
        self.set_rotation(*rotation)
        self.set_scale(*scale)

    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], self.translation[2])  # translate to GL loc ppint
        glRotatef(RAD2DEG * np.linalg.norm(self.rotation), self.rotation[0], self.rotation[1], self.rotation[2])
        glScalef(self.scale[0], self.scale[1], self.scale[2])

    def disable(self):
        glPopMatrix()

    def set_translation(self, newx, newy, newz):
        self.translation = (float(newx), float(newy), float(newz))

    def set_rotation(self, alpha, beta, gamma):
        # self.rotation = quaternion.from_euler_angles(alpha, beta, gamma)
        self.rotation = [alpha, beta, gamma]

    def set_scale(self, newx, newy, newz):
        self.scale = (float(newx), float(newy), float(newz))


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        glColor4f(*self.vec4)


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style

    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)

    def disable(self):
        glDisable(GL_LINE_STIPPLE)


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        glLineWidth(self.stroke)


class Sph(Geom):
    def __init__(self):
        Geom.__init__(self)
        self.batch = pyglet.graphics.Batch()

        self.do_batch()

    def do_batch(self):
        self.batch = pyglet.graphics.Batch()
        step = 10
        for lat in range(-90, 90, step):
            verts = []
            texc = []
            for lon in range(-180, 181, step):
                x = cos(radians(lat)) * cos(radians(lon))
                y = sin(radians(lat))
                z = cos(radians(lat)) * sin(radians(lon))
                # s = (lon + 180) / 360.0
                # t = (lat + 90) / 180.0
                verts += [x, y, z]
                # texc += [s, t]
                x = cos(radians(lat + step)) * cos(radians(lon))
                y = sin(radians(lat + step))
                z = cos(radians(lat + step)) * sin(radians(lon))
                # s = (lon + 180) / 360.0
                # t = ((lat + step) + 90) / 180.0
                verts += [x, y, z]
                # texc += [s, t]

            self.batch.add(len(verts) // 3, GL_TRIANGLE_STRIP, None, ('v3f', verts), ('c3B', len(verts)//3*[255,0,0]))

        points = list([0.9, 0, 0])
        points += list([1.1, 0, 0])
        self.batch.add(2, GL_LINE_STRIP, None, ('v3f', points), ('c3B', 2 * [90, 250, 200]))

    def render1(self):
        self.batch.draw()

def make_sphere(position, radius):
    sph = Sph()
    sph.add_attr(Transform(translation = position,scale = (radius,radius,radius)))
    return sph