from BodyMeasurements import BodyMeasurements


class Subject:
    def __init__(self):
        self.measurements = BodyMeasurements()
        self.height: float
        self.weight: float
        self.bodyStructure: str
        self.subject_id: str
        self.photo_id: str
