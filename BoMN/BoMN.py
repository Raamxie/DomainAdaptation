import Subject
import os


supportedImages = [".png"]
class BoMN(object):
    def __init__(self):
        self.dataset = list()
        self.images = dict()

    def openFolder(self, relativePath:str):
        with os.scandir(relativePath) as folderContents:
            for item in folderContents:
                if "." not in item:
                    self.openFolder(relativePath + "/" + item)
                    continue

                if item.split(".")[-1] in supportedImages:

