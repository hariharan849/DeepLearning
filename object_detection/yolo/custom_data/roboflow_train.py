import roboflow
from ultralytics import YOLO

# rf = roboflow.Roboflow(api_key="###")
# """
# Initializes the Roboflow object with the provided API key.
# """

# project = rf.workspace("learn-kop3c").project("apple-palm-pi40p")
# """
# Accesses the specified project within the workspace.
# """

# version = project.version(1)
# """
# Retrieves the specified version of the project.
# """

# dataset = version.download("yolov11")
# """
# Downloads the specified version of the dataset from Roboflow.
# """


rf = roboflow.Roboflow(api_key="6MwuxEBPFcyJT3HuFU9h")
project = rf.workspace("learn-kop3c").project("anpr-wpigd-1rlbq")
version = project.version(1)
dataset = version.download("yolov11")