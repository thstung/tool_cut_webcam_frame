import cv2
import numpy as np
from settings import (
    DIFFERENT_PERCENT,
    SOUNDBAR_COLORS,
    SOUNDBAR_EDGE,
    SOUNDBAR_LEFT,
    SOUNDBAR_MIN,
    SOUNDBAR_MAX,
    SOUNDBAR_PADDING,
    SOUNDBAR_TEMPLATE_DIR,
    SOUNDBAR_TOP,
    VOICE_INTERVAL_SECONDS,
)
from utilities import crop_soundbar_area


class VoiceProcessor:
    def __init__(self, class_config):
        self.voices = {}
        self.current_frame = set()
        self.current_interval = set()
        self.whole_video = []
        self.soundbar_count = []
        self.frame_count_in_an_interval = 0
        self.interval_frames = class_config.fps * VOICE_INTERVAL_SECONDS
        self.template = self.load_templates()

    def load_templates(self) -> dict:
        """
        Load color template of soundbar (yellow and green)
        """
        soundbar_template_dir = SOUNDBAR_TEMPLATE_DIR
        soundbar_colors = SOUNDBAR_COLORS
        templates = {}
        for color in soundbar_colors:
            templates[color] = cv2.imread(
                soundbar_template_dir + color + ".png",
                0,
            )

        return templates

    def update_totally_talking_time(self):
        """
        Calculate talking time of each person up to the current frame
        """
        if self.frame_count_in_an_interval <= self.interval_frames:
            for id in self.current_frame:
                if id not in self.current_interval:
                    self.current_interval.add(id)
            self.frame_count_in_an_interval += 1
        if (
            self.frame_count_in_an_interval == self.interval_frames
        ):  # after an interval
            for id in self.current_interval:
                self.whole_video[id] += self.interval_frames
            self.frame_count_in_an_interval = 0
            self.current_interval = set()
        # return self.voices, voice_frame_count

    def get_detected_voices(self, webcams: np.ndarray):
        """
        Get detected voices and their talking times
        """
        id = 0
        self.lst_soundbar_count = []
        self.current_frame = set()

        for id in range(len(webcams)):
            if id >= len(self.whole_video):
                self.whole_video.append(0)
            soundbar_area = crop_soundbar_area(webcams[id])
            total_soundbar = detect_talking_templateMatching(
                self.template,
                soundbar_area,
            )
            self.lst_soundbar_count.append(total_soundbar)
            if total_soundbar >= SOUNDBAR_MIN:
                self.current_frame.add(id)

        # estimate talking time
        self.update_totally_talking_time()

def select_template(templates, soundbar_frame):
    for template in templates:
        tem_a = np.average(templates[template])
        if (
            np.average(soundbar_frame[59:61, 5:7]) * 1.1 >= tem_a
            and np.average(soundbar_frame[59:61, 5:7]) * 0.9 <= tem_a
        ):
            return template
    return None


def detect_talking_templateMatching(templates, soundbar_area):
    soundbar_area = cv2.cvtColor(soundbar_area, cv2.COLOR_BGR2GRAY)
    color = select_template(templates, soundbar_area)
    soundbar_count = 0
    if color is None:
        return 0
    template = np.average(templates[color])
    for i in range(SOUNDBAR_MAX):
        x, y = (SOUNDBAR_LEFT, SOUNDBAR_PADDING + SOUNDBAR_TOP * i)
        imgCrop = soundbar_area[y : y + SOUNDBAR_EDGE, x : x + SOUNDBAR_EDGE]
        if (
            np.average(imgCrop) * (1 - DIFFERENT_PERCENT)
            < template
            < np.average(imgCrop) * (1 + DIFFERENT_PERCENT)
        ):
            soundbar_count += 1
    return soundbar_count
