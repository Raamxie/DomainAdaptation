from enum import Enum


class Measurement(Enum):
    HEAD = "Head Circumference"
    NECK = "Neck Circumference"
    SHOULDERSHOULDER = "Shoulder to shoulder"
    ARM = "Arm span"
    SHOULDERTOWRIST = "Shoulder to wrist"
    TORSO = "Torso length"
    BICEP = "Bicep circumference"
    WRIST = "Wrist circumference"
    CHEST = "Chest circumference"
    WAIST = "Waist circumference"
    PELVIS = "Pelvis circumference"
    LEG = "Leg length"
    INSEAM = "Inner leg length"
    THIGH = "Thigh circumference"
    KNEE = "Knee circumference"
    CALF = "Calf circumference"

class BodyMeasurements:
    def __init__(self):
        self.headCircumference: float
        self.neckCircumference: float
        self.shoulderToShoulder: float
        self.armSpan: float
        self.shoulderToWrist: float
        self.torsoLength: float
        self.bicepCircumference: float
        self.wristCircumference: float
        self.chestCircumference: float
        self.waistCircumference: float
        self.pelvisCircumference: float
        self.legLength: float
        self.innerLegLength: float
        self.thighCircumference: float
        self.kneeCircumference: float
        self.calfLength: float
