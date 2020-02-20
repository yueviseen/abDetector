import webbrowser
import datetime
import folium
import pymysql
import math
import time
import pandas as pd
import copy
from sklearn.cluster import DBSCAN, OPTICS, MeanShift
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import requests


class StayPoint:
    """
    The main purpose of this class is to detect the abnormal trajectory points
    """
    def __init__(self):
        # This variable mainly records and reads all trace points
        self.data = None

    def read_data(self, table):
        """
        Read data from database
        :param database: database
        :param table: table
        :return: None
        """

        # Connect to database
        development_conn = pymysql.connect(
            host="localhost",  # ip
            database="gps_jingchuang", # database
            user='root',  # user name
            password='yzw161112',  # password
            port=3344,  # port number
        )

        # Read all values of a table in the database
        read_sql = "select * from {0}".format(table)
        self.data = pd.read_sql(read_sql, development_conn)

        # Close database connection
        development_conn.close()
        print("read_data_success")

        return None

    def read_data_NOID(self, table):
        """
        Read data from database
        :param database: database
        :param table: table
        :return: None
        """

        # Connect to database
        development_conn = pymysql.connect(
            host="localhost",  # ip
            database="gps_jingchuang", # database
            user='root',  # user name
            password='yzw161112',  # password
            port=3344,  # port number
        )

        # Read all values of a table in the database
        read_sql = "select * from {0} where NO = '60' and VehicleID = '12511595'".format(table)
        self.data = pd.read_sql(read_sql, development_conn)

        # Close database connection
        development_conn.close()
        print("read_data_success")

        return None

    def write_data_road_intersection(self, table, data):
        """
        Write the data to the established database table
        :param database: database
        :param table: table
        :return: None
        """

        # Connect to database
        development_conn = pymysql.connect(
            host="localhost",  # ip
            database="gps_jingchuang", # database
            user='root',  # user name
            password='yzw161112',  # password
            port=3344,  # port number
        )

        # Write data to database tables in the form of strings
        for row in data:
            development_conn.cursor().execute("INSERT INTO {0} VALUES('{1}','{2}')".format(table, row[0], row[1]))
        development_conn.commit()

        # Connect to database
        development_conn.close()

        print("write_data_sp_success")

        return None

    def write_data_detour(self, table, data):
        """
        Write the data to the established database table
        :param database: database
        :param table: table
        :return: None
        """

        # Connect to database
        development_conn = pymysql.connect(
            host="localhost",  # ip
            database="gps_jingchuang", # database
            user='root',  # user name
            password='yzw161112',  # password
            port=3344,  # port number
        )

        # Write data to database tables in the form of strings
        for i in data:
            development_conn.cursor().execute("INSERT INTO {0} VALUES('{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}',"
                                              " '{9}', '{10}','{11}')".format(table, i[0], i[1], i[2], i[3], i[4], i[5],
                                                               i[6], i[7], i[8], i[9], i[10]))
        development_conn.commit()

        # Close database connection
        development_conn.close()

        print("write_data_detour_success")

        return None


def disjoint(x, y):
    """
    Find the trajectory near the x, y coordinates
    :param x: x coordinates
    :param y: y coordinates
    :return: Returns all associated values
    """
    mid = set()
    if dict_gird_trajectory.get((x, y)) is not None:
        mid = mid | dict_gird_trajectory[(x, y)]
    if dict_gird_trajectory.get((x, y + 1)) is not None:
        mid = mid | dict_gird_trajectory[(x, y + 1)]
    if dict_gird_trajectory.get((x, y - 1)) is not None:
        mid = mid | dict_gird_trajectory[(x, y - 1)]
    if dict_gird_trajectory.get((x + 1, y)) is not None:
        mid = mid | dict_gird_trajectory[(x + 1, y)]
    if dict_gird_trajectory.get((x - 1, y)) is not None:
        mid = mid | dict_gird_trajectory[(x - 1, y)]
    if dict_gird_trajectory.get((x - 1, y + 1)) is not None:
        mid = mid | dict_gird_trajectory[(x - 1, y + 1)]
    if dict_gird_trajectory.get((x + 1, y + 1)) is not None:
        mid = mid | dict_gird_trajectory[(x + 1, y + 1)]
    if dict_gird_trajectory.get((x + 1, y - 1)) is not None:
        mid = mid | dict_gird_trajectory[(x + 1, y - 1)]
    if dict_gird_trajectory.get((x - 1, y - 1)) is not None:
        mid = mid | dict_gird_trajectory[(x - 1, y - 1)]
    return mid


def score1(x):
    """
    Scoring rules
    :param x: Support x
    :return: Return score value
    """
    t = math.e
    tt = 150 * (x - 0.01)
    return 1/(1 + t**tt)


sp_invind = StayPoint()
sp_invind.read_data("trajectory_gird_auto_2_invind")
sp = StayPoint()
sp.read_data_NOID("gps_clearn_up_final_final_auto_detour")
print(len(sp_invind.data))
print(len(sp.data))
start = time.time()

# Inverted index dictionary
dict_gird_trajectory = {}
for index, row in sp_invind.data.iterrows():
    if dict_gird_trajectory.get((float(row[0]), float(row[1]))) is None:
        dict_gird_trajectory[(float(row[0]), float(row[1]))] = set()
    dict_gird_trajectory[(float(row[0]), float(row[1]))].add((float(row[2]), float(row[3])))

start1 = time.time()

"""
iBOAT algorithm
"""

# Adaptive window, false means the window size is 1
flag = False

# Record abnormal points
anomaly_point = []
T0 = 1291.0
size_window = []
threshold = 0.01
sc = 0

cnt = -1

# Traverse each trajectory point
lat_std = 0.003184870669716284
lng_std = 0.001935895047666416
for index, row in sp.data.iterrows():
    cnt = cnt + 1

    y = math.floor((float(row[4]) - 35.15090560913086) / lat_std)
    x = math.floor((float(row[3]) - 119.31890869140625) / lng_std)

    # Processing of the first trajectory point
    if flag is False and cnt == 0:
        mid = disjoint(x, y)
    elif flag is False and cnt > 1:
        mid1 = disjoint(x, y)
        tmp = mid1 & mid
        s = (len(tmp)-1) / len(mid)
        if threshold >= s:
            mid = mid1
            anomaly_point.append((x, y))
        else:
            mid = tmp
            flag = True
        mid_dis = float(sp.data.iloc[cnt][8]) - float(sp.data.iloc[cnt - 1][8])
        sc = sc + score1(s) * mid_dis
    else:
        mid1 = disjoint(x, y)
        tmp = mid & mid1
        s = (len(tmp)-1) / (len(mid))
        if threshold >= s:
            mid = mid1
            flag = False
            anomaly_point.append((x, y))
        else:
            mid = tmp
        mid_dis = float(sp.data.iloc[cnt][8]) - float(sp.data.iloc[cnt-1][8])
        sc = sc + score1(s) * mid_dis

print("time：")
print(time.time() - start1)
print(start1 - start)
print()
print(len(mid))
print(sc)
print(len(anomaly_point))