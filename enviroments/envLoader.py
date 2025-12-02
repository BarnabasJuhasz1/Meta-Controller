# lsd
from example_minigrid import SimpleEnv

#metra
from metraTester import MetraWrapper

# dads
from dads_minigrid import DadsWrapper  

# rsd
from minigrid_env import BaselineMiniGridEnv

# dyan
from minigrid_wrapper import MiniGridWrapper

# other envs

from AntMazeEnv import MazeWrapper

from maze_env import MazeEnv


class envLoader:
    def __init__(self,baseline="",env="minigrid"):

        if env =="minigrid":
            self.baseEnv=SimpleEnv()
            if baseline=="metra":
                self.env = MetraWrapper(self.baseEnv)
            elif baseline =="rsd":
                self.env = BaselineMiniGridEnv(self.baseEnv)
            elif baseline == "dads": 
                self.env = DadsWrapper(self.baseEnv)
            elif baseline== "dyan":
                self.env = MiniGridWrapper(self.baseEnv)
            elif baseline=="lsd":
                self.env = self.baseEnv
            else:
                print( "Wrong baseline name")
                return

        elif env == "maze":
            #self.baseEnv=MazeEnv()
            #self.env = MazeWrapper(self.baseEnv)
            pass
        





        
        return self.env

    