from pathlib import Path
import numpy as np
from gym import utils, spaces
from blendtorch import btt
from pyquaternion import Quaternion


class PregraspEnv(btt.env.OpenAIRemoteEnv):
    def __init__(self, render_every=10, real_time=False, image_width=200, image_height=200, show_obs=True, demo=False):
        super().__init__(version='0.0.1')
        self.launch(
            scene=Path(__file__).parent / 'pregrasp.blend',
            script=Path(__file__).parent / 'pregrasp.blend.py',
            real_time=real_time,
            render_every=10,
        )

        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.IMAGE_WIDTH_OUT = 96
        self.IMAGE_HEIGHT_OUT = 96

        # camera numberx3
        self.rotations = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0,
                          12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0}
        self.action_space_type = 'multidiscrete'
        self.step_called = 0

        self.show_observations = show_obs
        self.demo_mode = demo

        self.action_space = spaces.MultiDiscrete([self.IMAGE_WIDTH_OUT * self.IMAGE_HEIGHT_OUT, len(self.rotations)])
