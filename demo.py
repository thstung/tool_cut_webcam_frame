import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import (
    ImageDraw,
    ImageFont,
)
from settings import (
    DEVICE_KINDS,
    FONT_PATH,
    FONT_SIZE,
    WEBCAM_BOT,
    WEBCAM_LEFT_2,
    WEBCAM_LEFT_3,
    WEBCAM_PADDING,
    WEBCAM_TOP,
    WEBCAM_WIDTH,
)
from utilities import (
    convert_cv2_2_pil,
    convert_pil_2_cv2,
)
from video_config import ClassConfig
from voice_recognition import VoiceProcessor


sys.path.append(str(Path(__file__).resolve().parents[2] / "common/python"))
vietnamese_font = ImageFont.truetype(FONT_PATH, FONT_SIZE, encoding="utf-8")


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group("General")
    general.add_argument(
        "-i",
        "--input",
        required=True,
        help="Required. An input to process. The input must be a single image,"
        "a folder of images, video file or camera id.",
    )
    general.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Optional. Enable reading the input in a loop.",
    )
    general.add_argument(
        "-o",
        "--output",
        help="Optional. Name of the output file(s) to save.",
    )
    general.add_argument(
        "-limit",
        "--output_limit",
        default=0,
        type=int,
        help="Optional. Number of frames to store in output. "
        "If 0 is set, all frames are stored.",
    )
    # general.add_argument(
    #     "--output_resolution",
    #     default=None,
    #     type=resolution,
    #     help="Optional. Specify the maximum output window resolution "
    #     "in (width x height) format. Example: 1280x720. "
    #     "Input frame size used by default.",
    # )
    general.add_argument(
        "--no_show",
        action="store_true",
        help="Optional. Don't show output.",
    )
    general.add_argument(
        "--crop_size",
        default=(0, 0),
        type=int,
        nargs=2,
        help="Optional. Crop the input stream to this resolution.",
    )
    general.add_argument(
        "--match_algo",
        default="HUNGARIAN",
        choices=("HUNGARIAN", "MIN_DIST"),
        help="Optional. Algorithm for face matching. Default: HUNGARIAN.",
    )
    general.add_argument(
        "-u",
        "--utilization_monitors",
        default="",
        type=str,
        help="Optional. List of monitors to show initially.",
    )

    gallery = parser.add_argument_group("Faces database")
    gallery.add_argument(
        "-fg",
        default="",
        help="Optional. Path to the face images directory.",
    )
    gallery.add_argument(
        "--run_detector",
        action="store_true",
        help="Optional. Use Face Detection model to find faces "
        "on the face images, otherwise use full images.",
    )
    gallery.add_argument(
        "--allow_grow",
        action="store_true",
        help="Optional. Allow to grow faces gallery and to dump on disk. "
        "Available only if --no_show option is off.",
    )

    models = parser.add_argument_group("Models")

    models.add_argument(
        "--fd_input_size",
        default=(0, 0),
        type=int,
        nargs=2,
        help="Optional. Specify the input size of detection model for "
        "reshaping. Example: 500 700.",
    )

    infer = parser.add_argument_group("Inference options")
    infer.add_argument(
        "-d_fd",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Face Detection model. "
        "Default value is CPU.",
    )
    infer.add_argument(
        "-d_lm",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Facial Landmarks Detection "
        "model. Default value is CPU.",
    )
    infer.add_argument(
        "-d_reid",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Face Reidentification "
        "model. Default value is CPU.",
    )
    infer.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Optional. Be more verbose.",
    )
    infer.add_argument(
        "-t_fd",
        metavar="[0..1]",
        type=float,
        default=0.6,
        help="Optional. Probability threshold for face detections.",
    )
    infer.add_argument(
        "-t_id",
        metavar="[0..1]",
        type=float,
        default=0.3,
        help="Optional. Cosine distance threshold between two vectors "
        "for face identification.",
    )
    infer.add_argument(
        "-exp_r_fd",
        metavar="NUMBER",
        type=float,
        default=1.15,
        help="Optional. Scaling ratio for bboxes passed to face recognition.",
    )
    return parser

with open('name.txt', 'r') as f:
    data = f.read()
    name = data.split('\n')

def main():
    
    args = build_argparser().parse_args()

    class_config = ClassConfig(args)
    voice_processor = VoiceProcessor(class_config)
    # identity_processor = IdentityProcessor()
    frame_num = 0

    if args.output:
        height = int(class_config.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(class_config.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            class_config.fps,
            (width, height),
        )
        if video_writer.open is False:
            raise RuntimeError("Can't open video writer")

    # Start frame for Khanh_Minh: 1760, Thy: 15750
    frame_start = 1760
    frame_num = frame_start
    class_config.video.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    while class_config.video.isOpened():
        class_config.read_current_frame()

        if class_config.current_frame is None:
            if frame_num == 0:
                raise ValueError("Can't read an image from the input")
            break

        # Get number of people in the frame
        class_config.get_total_participants()

        # Get current webcam frame and position
        class_config.get_webcam_frame()

        # Detect talking persons
        voice_processor.get_detected_voices(class_config.webcam_frames)

        # Write ouput
        if args.output:
            frame = draw_detections(
                class_config,
                voice_processor,
            )
            video_writer.write(frame)
            cv2.imshow("Demo", frame)
            
            #save image webcam
            if cv2.waitKey(1) & 0xFF == ord("1"):
                save_webcam(0, class_config, frame, frame_num)
            if cv2.waitKey(1) & 0xFF == ord("2"):
                save_webcam(1, class_config, frame, frame_num)
            if cv2.waitKey(1) & 0xFF == ord("3"):
                save_webcam(2, class_config, frame, frame_num)
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


        frame_num += 1

def save_webcam(id, class_config, output_frame, frame_num):
    wc_padding = WEBCAM_PADDING[class_config.type]
    wc_width = WEBCAM_WIDTH[class_config.type]
    wc_top = WEBCAM_TOP[class_config.type]
    wc_bot = WEBCAM_BOT[class_config.type]
    wc_left = WEBCAM_LEFT_2[class_config.type] + (id * (wc_padding + wc_width))  # for people = 2
    if class_config.participants == 3:
        wc_left = WEBCAM_LEFT_3[class_config.type] + (id * (wc_padding + wc_width))
    wc_right = wc_left + wc_width
    web_frame = output_frame[wc_top:wc_bot, wc_left:wc_right]
    time = str(int(frame_num/class_config.fps)//60) + str(int(frame_num/class_config.fps)%60)
    cv2.imwrite(f"output/{name[id] + time}.png", web_frame)

def draw_detections(
    class_config: ClassConfig,
    # identity_processor: IdentityProcessor,
    voice_processor: VoiceProcessor,
) -> np.ndarray:
    """
    Method to draw detection of the following things:
    - Voice detection
    """
    output_frame = class_config.current_frame
    output_frame = draw_voice_detection(
        class_config,
        voice_processor,
        output_frame,
    )
    return output_frame


def draw_voice_detection(
    class_config: ClassConfig,
    voice_processor: VoiceProcessor,
    output_frame: np.ndarray,
) -> np.ndarray:
    """
    Draw bounding boxes around the webcam of a talking person
    """
    for i in range(len(class_config.webcam_frames)):
        org = (class_config.webcam_positions[i][2] + 30, class_config.webcam_positions[i][0] + 30)
        output_frame = cv2.putText(
            output_frame, 
            str(voice_processor.lst_soundbar_count[i]), 
            org, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 0, 0), 
            2, 
            cv2.LINE_AA)

    for i in voice_processor.current_frame:
        # Additional numbers are for fitting the webcam's bounding box
        top = class_config.webcam_positions[i][0] + 5
        bot = class_config.webcam_positions[i][1] - 3
        left = class_config.webcam_positions[i][2]
        right = class_config.webcam_positions[i][3] - 2
        cv2.rectangle(
            output_frame,
            (left, top),
            (right, bot),
            (255, 0, 0),
            2,
        )
        org = (class_config.webcam_positions[i][2] + 30, class_config.webcam_positions[i][0] + 30)
        output_frame = cv2.putText(
            output_frame, 
            str(voice_processor.lst_soundbar_count[i]), 
            org, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 0, 0), 
            2, 
            cv2.LINE_AA)
    return output_frame

def write_recognized_name(
    output_frame: np.ndarray,
    text: str,
    xmin: int,
    ymin: int,
) -> np.ndarray:
    """write name and confident score of person"""
    # this method can write utf-8 text

    text_width, text_height = vietnamese_font.getsize(text)

    cv2.rectangle(
        output_frame,
        (xmin, ymin),
        (xmin + text_width, ymin - text_height),
        (255, 255, 0),
        cv2.FILLED,
    )

    img_pil = convert_cv2_2_pil(output_frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((xmin, ymin - text_height), text, "black", vietnamese_font)
    return convert_pil_2_cv2(img_pil)


def add_audio2video(args):
    # def convert_video_to_audio_ffmpeg(video_file, output_ext="mp3"):
    """
    Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module
    """
    # Read videos
    output_video = VideoFileClip(args.output)
    original_video = VideoFileClip(args.input)

    # Extract duration
    output_duration = output_video.duration
    original_duration = original_video.duration

    # Trim original audio
    trimmed_video = original_video.subclip(
        original_duration - output_duration,
        original_duration,
    )
    print(f"{original_duration} - {output_duration}")

    # Extract audio from trimmed video
    trimmed_audio = trimmed_video.audio

    # Add audio to output video
    output_video = output_video.set_audio(trimmed_audio)

    # Write audio with audio
    output_video.write_videofile(args.output)


if __name__ == "__main__":
    sys.exit(main() or 0)
