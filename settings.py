# List of devices can be used for training/recognizing
DEVICE_KINDS = ["CPU", "GPU", "MYRIAD", "HETERO", "HDDL"]

# Models
MODEL_FACE_DETECTION = (
    "models/face-detection-retail-0004/FP16/face-detection-retail-0004.xml"
)
MODEL_FACE_REIDENTIFICATION = "models/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml"  # noqa: E501
MODEL_LANDMARKS = "models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml"  # noqa: E501
OCR_MODELS = "models/vietocr/vietocr.model"


# Face settings:
IMAGE_EXTENSIONS = ["jpg", "png"]
FACE_INTERVAL_SECONDS = 3
REGISTERED_FACE = "registered_faces/"

# Face_Landmark
POINTS_NUMBER = 5


# Voice settings:
DIFFERENT_PERCENT = 0.05
SOUNDBAR_TEMPLATE_DIR = "soundbar_template/"
SOUNDBAR_COLORS = ["green", "yellow"]
SOUNDBAR_MAX = 15
SOUNDBAR_MIN = 4
SOUNDBAR_EDGE = 2
SOUNDBAR_LEFT = 5
SOUNDBAR_TOP = 4
SOUNDBAR_PADDING = 3
VOICE_INTERVAL_SECONDS = 1


# Name settings:
FONT_SIZE = 15
FONT_PATH = "font/Roboto-Regular.ttf"  # Vietnamese Font


# OCR
CLASS_DURATION = 31
OCR_NETWORK = "vgg_transformer"
OCR_DEVICE = "cpu"  # gpu:0

# Area
# [[y, y + height], [x, x + width]]
BACKGROUND_MEAN_COLOR = 210.47
BACKGROUND_MEAN_COLOR_DIF = 0.01  # 1%
WEBCAM_TOP = [20, 30]
WEBCAM_BOT = [130, 195]
WEBCAM_LEFT_2 = [460, 690]  # for people = 2
WEBCAM_LEFT_3 = [369, 534]  # for people = 3
WEBCAM_PADDING = [2, 3]
WEBCAM_WIDTH = [180, 270]
WEBCAM_MID = 400

# Checking existing webcam
WEBCAM_CHECKING_BOT = 125
WEBCAM_CHECKING_LEFT_2 = 3
WEBCAM_CHECKING_LEFT_3 = 2
WEBCAM_CHECKING_TOP = 100
WEBCAM_CHECKING_WIDTH = 2

# Crop from the whole frame
CLASS_STARTING_AREA = [[5, 20], [600, 850]]

# Crop from webcam area
FRAME_MICRO = [[90, 105], [3, 13]]
FRAME_NAME = [[90, 105], [13, 150]]
FRAME_SOUND_LVL = [[30, 91], [3, 13]]

# Crop from sound_lvl
TEMPLATE_COMPARATION = [[59, 61], [4, 8]]
