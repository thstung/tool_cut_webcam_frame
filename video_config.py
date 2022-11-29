import re
import cv2
from settings import (
    BACKGROUND_MEAN_COLOR,
    BACKGROUND_MEAN_COLOR_DIF,
    WEBCAM_BOT,
    WEBCAM_CHECKING_BOT,
    WEBCAM_CHECKING_LEFT_2,
    WEBCAM_CHECKING_LEFT_3,
    WEBCAM_CHECKING_TOP,
    WEBCAM_CHECKING_WIDTH,
    WEBCAM_LEFT_2,
    WEBCAM_LEFT_3,
    WEBCAM_MID,
    WEBCAM_PADDING,
    WEBCAM_TOP,
    WEBCAM_WIDTH,
)
from utilities import convert_cv2_2_pil


class ClassConfig:
    def __init__(self, args):
        self.video = cv2.VideoCapture(args.input)
        if self.video.open is False:
            raise RuntimeError("Can't open video. Please check input.")
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 1)
        frame = self.video.read()[1]
        self.type = 0 if frame.shape[0] == 720 else 1
        self.class_start_flag = False
        self.class_duration = 0

        self.frame_start = 0
        self.frame_class = 0

        self.updated_participants = 0
        self.participants = 0
        self.participants_change = 0

        self.current_frame = []
        self.webcam_frames = []
        self.webcam_positions = []

    def read_current_frame(self):
        """
        Method for reading a frame in a video
        """
        _, self.current_frame = self.video.read()

    def get_webcam_frame(self):
        """
        Extract frames of webcams from a given number of participants
        """
        wc_padding = WEBCAM_PADDING[self.type]
        wc_width = WEBCAM_WIDTH[self.type]
        wc_top = WEBCAM_TOP[self.type]
        wc_bot = WEBCAM_BOT[self.type]
        wc_left = WEBCAM_LEFT_2[self.type]  # for people = 2
        if self.participants == 3:
            wc_left = WEBCAM_LEFT_3[self.type]

        self.webcam_frames = []
        for each in range(0, self.participants):
            wc_right = wc_left + wc_width
            web_frame = self.current_frame[wc_top:wc_bot, wc_left:wc_right]
            self.webcam_frames.append(web_frame)
            wc_left = wc_right + wc_padding

        if self.participants_change != 0:
            self.webcam_positions = []
            wc_left = WEBCAM_LEFT_2[self.type]  # for people = 2
            if self.participants == 3:
                wc_left = WEBCAM_LEFT_3[self.type]
            for each in range(0, self.participants):
                wc_right = wc_left + wc_width
                web_frame = self.current_frame[wc_top:wc_bot, wc_left:wc_right]
                self.webcam_positions.append(
                    [
                        WEBCAM_TOP[self.type],
                        WEBCAM_BOT[self.type],
                        wc_left,
                        wc_right,
                    ],
                )
                wc_left = wc_right + wc_padding

    def get_total_participants(self) -> int:
        """
        Get number of persons in the current frame.
        Update the change of that number (comparing to the previous frame)
        """
        bg = self.current_frame[
            WEBCAM_TOP[self.type]:WEBCAM_BOT[self.type],
            WEBCAM_LEFT_3[self.type]:WEBCAM_MID,
        ]
        if (
            abs(bg.mean() - BACKGROUND_MEAN_COLOR)
            < BACKGROUND_MEAN_COLOR * BACKGROUND_MEAN_COLOR_DIF
        ):
            self.updated_participants = 2
        else:
            self.updated_participants = 3

        if self.updated_participants != self.participants:
            self.participants_change = (
                self.updated_participants - self.participants
            )
            self.participants = self.updated_participants
        else:
            self.participants_change = 0

    def check_existing_webcam(
        self,
        participant_id: int,
    ) -> bool:
        """
        Check if a given webcam is in the correct position
        Assume that webcam #1 is always in the correct position
        """

        if participant_id == 0:
            return True

        """ Case: 2 webcams
        Order of webcam is counted from left to right
        apply for webcam #1"""
        if self.participants == 2:  # webcam #2
            if participant_id == 1:
                left_area = (
                    WEBCAM_LEFT_2[self.type] + WEBCAM_WIDTH[self.type] + WEBCAM_CHECKING_LEFT_2
                )  # noqa: E501
                check_area = self.current_frame[
                    WEBCAM_CHECKING_TOP:WEBCAM_CHECKING_BOT,
                    left_area : left_area + WEBCAM_CHECKING_WIDTH,
                ]
                if (
                    abs(check_area.mean() - BACKGROUND_MEAN_COLOR)
                    < BACKGROUND_MEAN_COLOR * BACKGROUND_MEAN_COLOR_DIF
                ):
                    return False
                return True

        """ Case: 3 webcams
        Order of webcam is counted from left to right
        Apply for webcam #2 & #3"""
        if self.participants == 3:
            left_area = (
                WEBCAM_LEFT_3[self.type]
                + (WEBCAM_WIDTH + WEBCAM_CHECKING_LEFT_3) * participant_id
            )  # noqa: E501
            check_area = self.current_frame[
                WEBCAM_CHECKING_TOP:WEBCAM_CHECKING_BOT,
                left_area : left_area + WEBCAM_CHECKING_WIDTH,
            ]
            if (
                abs(check_area.mean() - BACKGROUND_MEAN_COLOR)
                < BACKGROUND_MEAN_COLOR * BACKGROUND_MEAN_COLOR_DIF
            ):
                return False
            return True

        return False
