import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC
from datetime import datetime
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from tqdm import tqdm

from model_config import Config
from .base_model import BaseModel


def _init_dirs(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except:
            pass


class Trainer(BaseModel):
    def __init__(self, conf):
        super().__init__(conf)
        self.sensor_name = conf.SENSOR_NAME
        self.directory = f"{Config.MODEL_OUT}/{Config.NAME}"
        self.file_path = f"{self.directory}/{self.sensor_name}.h5"

        _init_dirs(self.directory)

    def train(self):
        self.data_load()
        open(self.file_path, 'w').close()

    def data_load(self):
        """
        데이터 로드는 Config.DATA_PATH의 파일들 중 self.sensor_name이 포함된 파일들을 불러오고,
        이름 순으로 정렬 후, DF로 합치는 방식으로 구현 가능
        """
        pass
