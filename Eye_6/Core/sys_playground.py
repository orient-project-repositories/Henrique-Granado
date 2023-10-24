from pyglet.window import key
import tkinter as tk
import threading
import ctypes
import numpy as np
import quaternion
from Core.System import *

class MotorApp(tk.Frame):
    def __init__(self, master, objects):
        tk.Frame.__init__(self, master)
        self.master = master
        self.model = objects[0]
        self.window = objects[1]
        self.check = tk.BooleanVar()
        self.time_check = tk.BooleanVar()

        self.frame1 = tk.Frame(self, bd= 1, relief = "raised")
        self.frame2 = tk.Frame(self, bd= 1, relief = "raised")

        self.timestep_label = tk.Label(self.frame1, text="Time Step")
        self.timestep_slider = tk.Scale(self.frame1, from_=0, to=6, orient=tk.HORIZONTAL, command = self.set_timer_value, showvalue = 0)
        self.timestep_text = tk.Label(self.frame1, text = "0.00001")
        self.timestep_checkbox = tk.Checkbutton(self.frame1, variable=self.time_check, text="Real Time", command=self.time_checking)
        self.motor1_label = tk.Label(self.frame2, text="Motor 1")
        self.motor1_slider = tk.Scale(self.frame2, from_=-60, to=60, orient=tk.HORIZONTAL)
        self.motor2_label = tk.Label(self.frame2, text="Motor 2")
        self.motor2_slider = tk.Scale(self.frame2, from_=-60, to=60, orient=tk.HORIZONTAL)
        self.motor3_label = tk.Label(self.frame2, text="Motor 3")
        self.motor3_slider = tk.Scale(self.frame2, from_=-60, to=60, orient=tk.HORIZONTAL)
        self.motor4_label = tk.Label(self.frame2, text="Motor 4")
        self.motor4_slider = tk.Scale(self.frame2, from_=-60, to=60, orient=tk.HORIZONTAL)
        self.motor5_label = tk.Label(self.frame2, text="Motor 5")
        self.motor5_slider = tk.Scale(self.frame2, from_=-60, to=60, orient=tk.HORIZONTAL)
        self.motor6_label = tk.Label(self.frame2, text="Motor 6")
        self.motor6_slider = tk.Scale(self.frame2, from_=-60, to=60, orient=tk.HORIZONTAL)
        self.update_button = tk.Button(self.frame2, text="Get Update", command=self.button_pressed)
        self.auto_update = tk.Checkbutton(self.frame2, variable=self.check, text="Autoupdate", command=self.checking)

        self.widgets = [self.motor1_label, self.motor1_slider, self.motor2_label, self.motor2_slider, self.motor3_label, self.motor3_slider,
                        self.motor4_label, self.motor4_slider, self.update_button, self.auto_update]


        self.pack(padx = 5, pady = 5)
        self.frame1.pack(fill = "both", expand = True)
        self.frame2.pack(fill = "both", expand = True)
        self.set_timer_value(0)

        self.update_widgets()

        self.master.bind("<ButtonRelease-1>", self.click_event)

    def set_timer_value(self, val):
        values = [0.00001, 0.00005, 0.0001,  0.0005, 0.001,  0.005, 0.01]
        str_values = ["{:.5f}".format(x) if x < 0.0001 else "{}".format(x) for x in values]
        self.timestep_text.configure(text='{} seconds'.format(str_values[int(val)]))
        self.time_update()

    def time_checking(self):
        if self.time_check.get():
            self.timestep_slider['state'] = tk.DISABLED
            # self.timestep_text['relief'] = "sunken"
        else:
            self.timestep_slider['state'] = tk.NORMAL
            # self.timestep_text['relief'] = "flat"
        self.time_update()

    def time_update(self):
        if self.time_check.get():
            print(self.time_check.get())
            self.window.set_timestep(10)
        else:
            values = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
            n_dt = values[int(self.timestep_slider.get())]
            self.window.set_timestep(n_dt)


    def update_widgets(self):
        for w in self.widgets:
            self.grid_forget()
        self.motor1_label.grid(row=0, column=0, sticky = "s")
        self.motor1_slider.grid(row=0, column=1)
        self.motor2_label.grid(row=1, column=0, sticky = "s")
        self.motor2_slider.grid(row=1, column=1)
        self.motor3_label.grid(row=2, column=0, sticky = "s")
        self.motor3_slider.grid(row=2, column=1)
        self.motor4_label.grid(row=3, column=0, sticky = "s")
        self.motor4_slider.grid(row=3, column=1)
        self.motor5_label.grid(row=4, column=0, sticky = "s")
        self.motor5_slider.grid(row=4, column=1)
        self.motor6_label.grid(row=5, column=0, sticky = "s")
        self.motor6_slider.grid(row=5, column=1)
        self.auto_update.grid(row=6, column=0)
        self.update_button.grid(row=6, column=1)
        self.timestep_label.grid(row = 0, column = 0)
        self.timestep_slider.grid(row=0, column=1)
        self.timestep_text.grid(row=1, column=1, sticky = "n")
        self.timestep_checkbox.grid(row = 1 , column = 0)

    def checking(self):
        if self.check.get():
            self.update_button['state'] = tk.DISABLED
        else:
            self.update_button['state'] = tk.NORMAL

    def button_pressed(self):
        print("Button Pressed")
        self.update_model()

    def update_model(self):
        stuff = np.radians([self.motor1_slider.get(), self.motor2_slider.get(), self.motor3_slider.get(), self.motor4_slider.get(), self.motor5_slider.get(), self.motor6_slider.get()])
        self.model.feed_input(stuff)

    def click_event(self, event):
        # print("Clicked")
        if self.check.get():
            self.update_model()


class Player:
    def __init__(self,pos=(0,0,0),rot=(0,0)):
        self.pos = list(pos)
        self.rot = list(rot)

    def mouse_motion(self,dx,dy):
        dx/=8; dy/=8; self.rot[0]+=dy; self.rot[1]-=dx
        if self.rot[0]>90: self.rot[0] = 90
        elif self.rot[0]<-90: self.rot[0] = -90

    def update(self,dt,keys):
        mult = 1
        if keys[key.LCTRL]:
            mult = 0.1

        s = dt*10*mult
        rotY = -self.rot[1]/180*math.pi
        dx,dz = s*math.sin(rotY)*mult,s*math.cos(rotY)*mult
        if keys[key.W]: self.pos[0]+=dx; self.pos[2]-=dz
        if keys[key.S]: self.pos[0]-=dx; self.pos[2]+=dz
        if keys[key.A]: self.pos[0]-=dz; self.pos[2]-=dx
        if keys[key.D]: self.pos[0]+=dz; self.pos[2]+=dx

        if keys[key.SPACE]: self.pos[1]+=s
        if keys[key.LSHIFT]: self.pos[1]-=s


class Window(pyglet.window.Window):

    def push(self,pos,rot): glPushMatrix(); glRotatef(-rot[0],1,0,0); glRotatef(-rot[1],0,1,0); glTranslatef(-pos[0],-pos[1],-pos[2],)
    def Projection(self): glMatrixMode(GL_PROJECTION); glLoadIdentity()
    def Model(self): glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    def set2d(self): self.Projection(); gluOrtho2D(0,self.width,0,self.height); self.Model()
    def set3d(self): self.Projection(); gluPerspective(90,self.width/self.height,0.05,1000); self.Model()

    def setLock(self,state): self.lock = state; self.set_exclusive_mouse(state)
    lock = False; mouse_lock = property(lambda self:self.lock, setLock)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(300,200)
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self.timestep = 0.00001
        pyglet.clock.schedule_interval(self.update, 0.005)

        self.model_list = []
        self.player = Player([0, 0, 4.25], [0, 0])# Player([-17.045598021490832, 14.155081999999988, -8.519757164101744], [-17.25, 217.375])  # Player((-2.5, 0, -1.5),(0, -120))

        self.default_setup()

    def set_timestep(self, v):
        self.timestep = v


    def default_setup(self):
        glClearColor(0.5, 0.7, 1, 1)
        glEnable(GL_DEPTH_TEST)
        # glEnable(GL_CULL_FACE)
        self.lighting_setup()
        self.graphic_properties_setup()

    def lighting_setup(self):
        # LIGHTING
        glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(1, 1, 1, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(1, 1, 1, 1))
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(0, 0, 0, 0))
        glEnable(GL_LIGHT0)

        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glShadeModel(GL_SMOOTH)

        glMaterialfv(GL_FRONT, GL_AMBIENT, _gl_vector(0.192250, 0.192250, 0.192250))
        glMaterialfv(GL_FRONT, GL_DIFFUSE, _gl_vector(0.507540, 0.507540, 0.507540))
        glMaterialfv(GL_FRONT, GL_SPECULAR, _gl_vector(.5082730, .5082730, .5082730))

        glMaterialf(GL_FRONT, GL_SHININESS, .1 * 128.0);

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def graphic_properties_setup(self):
        # LINE THICKNESS
        glLineWidth(4)
        glPointSize(10)

    def on_mouse_motion(self,x,y,dx,dy):
        if self.mouse_lock: self.player.mouse_motion(dx, dy)

    def on_key_press(self, KEY, MOD):
        if KEY == key.ESCAPE: self.close()
        elif KEY == key.E: self.mouse_lock = not self.mouse_lock
        elif KEY == key.I:
            print(self.player.pos, self.player.rot)
        elif KEY == key.J:
            print(self.model_list[0].get_output())

    def update(self, dt):
        # print(dt)
        t_step = dt*int(dt<=self.timestep)+self.timestep*int(dt>self.timestep)
        self.player.update(dt, self.keys)
        self.definitely_not_update(t_step)

    def definitely_not_update(self, dt):
        for model in self.model_list:
            model.update(dt)
        # self.sphere.update()

    def on_draw(self):
        self.clear()
        self.switch_to()
        self.dispatch_events()
        self.set3d()
        self.push(self.player.pos, self.player.rot)
        for model in self.model_list:
            model.render(False)
        glPopMatrix()

    def add_model(self, model):
        self.model_list.append(model)


def thread(model):
    master = tk.Tk()
    app = MotorApp(master, model)
    app.mainloop()
    while True:
        pass


def float_ctype_array (*args):
    return (ctypes.c_float * len(args))(*args)


def _gl_vector(array, *args):
    '''
    Convert an array and an optional set of args into a flat vector of GLfloat
    '''
    array = np.array(array)
    if len(args) > 0:
        array = np.append(array, args)
    vector = (GLfloat * len(array))(*array)
    return vector

if __name__ == '__main__':
    floor = Floor()

    system = System()
    window = Window(width=854,height=480,caption='EYE',resizable=True, vsync = False)
    window.add_model(system)
    window.add_model(floor)

    shared_variables = [system, window]

    thread_thread = threading.Thread(target=thread, args=(shared_variables,))
    thread_thread.setDaemon(True)
    thread_thread.start()

    pyglet.app.run()

