from roboflow import Roboflow
rf = Roboflow(api_key="0qEjha6CKkiHJsmrBZPX")
project = rf.workspace("robotics-pxqqp").project("ai-4-robotics")
version = project.version(1)
dataset = version.download("yolov8")
