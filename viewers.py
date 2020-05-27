import requests
import json
import matplotlib.pyplot as plt
import dateutil
import datetime
import pandas as pd
import numpy as np
from urllib.parse import urljoin
from collections import defaultdict

root_url = "http://staging-dot-ccnr-michael-danziger.appspot.com/"


def parse(string_value):
    try:
        return float(string_value)
    except:
        try:
            return dateutil.parser.parse(string_value)
        except:
            return string_value


def isintlike(x):
    if isinstance(x, int):
        return True
    if isinstance(x, str) and x.isnumeric():
        return True
    if isinstance(x, float) and x.is_integer():
        return True
    return False


class Hospital(object):

    def __init__(self, hospital_id, metadata_dict=None):
        self.hospital_id = str(hospital_id)
        self.root_url = root_url
        if metadata_dict is None:
            self.load_metadata()
        else:
            for k, v in metadata_dict.items():
                self.__dict__[k] = v

    def load_metadata(self):
        url = f"/api/hospital/{self.hospital_id}/metadata/"
        url = urljoin(self.root_url, url)
        r = requests.get(url)
        for k, v in r.json().items():
            self.__dict__[k] = v

    def get_raw_history(self):
        self.raw_history = {}
        url = f"/api/hospital/{self.hospital_id}/history/"
        url = urljoin(self.root_url, url)
        r = requests.get(url)
        for k, v in r.json().items():
            self.raw_history[k] = [parse(i) for i in v]

    def get_hourly_history(self):
        self.hourly_history = {}
        url = f"/api/hospital/{self.hospital_id}/hourly/"
        url = urljoin(self.root_url, url)
        r = requests.get(url)
        for k, v in r.json().items():
            self.hourly_history[k] = [parse(i) for i in v]

    def get_daily_history(self):
        self.daily_history = {}
        url = f"/api/hospital/{self.hospital_id}/daily/"
        url = urljoin(self.root_url, url)
        r = requests.get(url)
        for k, v in r.json().items():
            self.daily_history[k] = [parse(i) for i in v]


class HospitalSet(object):

    def __init__(self, hospital_list=None):
        self.root_url = root_url
        self.hospitals = []
        if hospital_list is None:
            return
        if all(map(isintlike, hospital_list)):
            self.hospitals = [Hospital(i) for i in hospital_list]
        elif all(map(lambda x: isinstance(x, Hospital), self.hospitals)):
            self.hospitals = hospital_list

    def get_all_hospitals(self):
        url = "/api/hospitals/"
        url = urljoin(self.root_url, url)
        r = requests.get(url)
        self.hospitals = [Hospital(x["hospital_id"], x) for x in r.json()]

    def get_all_history(self,freq="daily"):
        if freq == "daily":
            url = "/api/dump/daily/"
        elif freq == "hourly":
            url = "/api/dump/hourly/"
        else:
            raise NotImplementedError("Frequency not supported")
        url = urljoin(self.root_url, url)
        r = requests.get(url)
        all_histories = defaultdict(lambda: {"observed_time": [], "wait_time_mins": []})
        for row in r.json():
            all_histories[row["hospital_id"]]["observed_time"].append(parse(row["observed_time"]))
            all_histories[row["hospital_id"]]["wait_time_mins"].append(parse(row["wait_time_mins"]))
        for h in self.hospitals:
            h.__dict__[f"{freq}_history"] = all_histories[h.hospital_id]

    def generate_matrix_from_histories(self,freq="daily"):

        latest_time = datetime.datetime(1970, 1, 1,tzinfo=datetime.timezone.utc)
        earliest_time = datetime.datetime.now(datetime.timezone.utc)
        for h in self.hospitals:
            if f"{freq}_history" in h.__dict__ and h.__dict__[f"{freq}_history"]["observed_time"]:
                earliest_time = min(earliest_time, min(h.__dict__[f"{freq}_history"]["observed_time"]))
                latest_time = max(latest_time, max(h.__dict__[f"{freq}_history"]["observed_time"]))
        time_index = []
        time_step = datetime.timedelta(days=1) if freq == "daily" else datetime.timedelta(hours=1)
        current_time = earliest_time
        while current_time <= latest_time:
            time_index.append(current_time.isoformat())
            current_time += time_step
        time_index_dict = dict((v,idx) for idx,v in enumerate(time_index))
        if not "matrix" in self.__dict__:
            self.matrix = {}
            self.time_index = {}
            self.df = {}
        self.matrix[freq] = -np.ones((len(self.hospitals),len(time_index) ))
        for hid,h in enumerate(self.hospitals):
            d = h.__dict__[f"{freq}_history"]
            for t,v in zip(d["observed_time"],d["wait_time_mins"]):
                self.matrix[freq][hid, time_index_dict[t.isoformat()]] = v
        self.time_index[freq] = [dateutil.parser.parse(t) for t in time_index]
        self.matrix[freq] = np.ma.masked_less(self.matrix[freq],0)
        self.df[freq] = pd.DataFrame(self.matrix[freq], index=[h.hospital_name for h in self.hospitals],
                          columns=self.time_index[freq])

    def get_matrix_view(self,freq="daily",date_filter=None,hospital_filter=None):
        if date_filter is None:
            date_filter = lambda x: True
        if hospital_filter is None:
            hospital_filter = lambda x: True
        tidx = [True if date_filter(t) else False for i,t in enumerate(self.time_index[freq])]
        hidx = [True if hospital_filter(h) else False for i,h in enumerate(self.hospitals)]
        return self.matrix[freq][:,tidx][hidx,:]


    def plot_hospital_matrix(self,freq="daily"):
        plt.pcolormesh([t[:10] for t in self.time_index[freq]], [h.hospital_name for h in self.hospitals],
                   self.matrix[freq])
        plt.xticks(rotation=50)

    def get_impacted_fraction(self,freq="daily",date_filter=None,threshold=15):
        df = self.df[freq]
        if date_filter is None:
            date_filter = lambda x: True
        df_frac = (df[df >= threshold].count() / df[df >= 0].count()).reset_index()
        df_frac = df_frac[df_frac["index"].apply(date_filter)]
        x, y = df_frac.values.T
        if isinstance(x[0],str):
            x = [dateutil.parser.parse(i).date() for i in x]
        return x,y

    def plot_impacted_fraction(self,freq="daily",threshold=15):
        x,y=self.get_impacted_fraction(freq,threshold)
        plt.plot_date(x, y)
        plt.xticks(rotation=50)

    def get_urgent_cares(self):
        return HospitalSet(list(filter(lambda x: x.type_id == '2', self.hospitals)))

    def get_ers(self):
        return HospitalSet(list(filter(lambda x: x.type_id == '1', self.hospitals)))

    def get_in_state(self,state):
        return HospitalSet(list(filter(lambda x: x.state == state, self.hospitals)))
