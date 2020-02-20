import webbrowser
import datetime
import math
import folium
import pymysql
import time
import pandas as pd
import numpy as np
import copy
from sklearn.cluster import DBSCAN, OPTICS, MeanShift
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import requests


def get_distance(a, b):
    """
    Calculate the distance between two trajectory points
    :param a: trajectory point a
    :param b: trajectory point b
    :return: the distance between two trajectory points（m）
    """
    return geodesic((a[2], a[3]), (b[2], b[3])).meters


class StayPoint:
    """
    The main purpose of this class is to detect the abnormal trajectory points
    """
    def __init__(self):
        # This variable mainly records and reads all trace points
        self.data = None

        # Set the time span of stay points
        self.length_of_time = 1800

        # Set the speed span of stay points
        self.length_of_veo = 5

        # record clustering result data
        self.data_cluster = None

        # Record the data of the stay point
        self.sp_data = None

        # Record the train number of multiple orders
        self.data_mul_order = None

        # Record the end point of each train
        self.data_car_end = None

        # Record the time and distance between each cluster
        self.data_time_distance = None

    def read_data(self, table):
        """
        Read data from database
        :param database: database name
        :param table: table name
        :return: None
        """

        # Connect to database
        development_conn = pymysql.connect(
            host="localhost",  # ip
            database="gps_jingchuang", #database
            user='root',  # user name
            password='yzw161112',  # password
            port=3344,  # port nuber
        )

        # Read all values of a table in the database
        read_sql = "select * from {0}".format(table)
        self.data = pd.read_sql(read_sql, development_conn)

        # Close database connection
        development_conn.close()
        print("read_data_success")

        return None

    def read_data_sp(self, table):
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
        self.sp_data = pd.read_sql(read_sql, development_conn)

        # Close database connection
        development_conn.close()

        print("read_data_sp_success")
        return None

    def read_data_cluster(self, table):
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
        self.data_cluster = pd.read_sql(read_sql, development_conn)

        # Close database connection
        development_conn.close()

        print("read_data_cluster_success")
        return None

    def write_data_sp(self, table):
        """
        Write the data to the established database table
        :param database: 数据库
        :param table: 表名
        :return: None
        """

        # Connect to database
        development_conn = pymysql.connect(
            host="localhost",  # ip
            database="gps_jingchuang",
            user='root',  # 用户名
            password='yzw161112',  # 密码
            port=3344,  # 端口号
        )

        # Write data to database tables in the form of strings
        for index, row in self.sp_data.iterrows():
            development_conn.cursor().execute("INSERT INTO {0} VALUES('{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}',"
                                              " '{9}')".format(table, row[0], row[1], row[2], row[3], row[4], row[5],
                                                               row[6], row[7], row[8]))
        development_conn.commit()

        # Close database connection
        development_conn.close()

        print("write_data_sp_success")

        return None

    def write_data_cluster(self, table):
        """
        Write the data to the established database table
        :param table: 表名
        :return: None
        """

        # Connect to database
        development_conn = pymysql.connect(
            host="localhost",  # ip
            database="gps_jingchuang",# database
            user='root',  # user name
            password='yzw161112',  # password
            port=3344,  # port number
        )

        # Write data to database tables in the form of strings
        for index, row in self.data_cluster.iterrows():
            print(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row["y"])
            development_conn.cursor().execute("INSERT INTO {0} VALUES('{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}',"
                                              " '{9}','{10}')".format(table, row[0], row[1], row[2], row[3], row[4],
                                                                      row[5], row[6], row[7], row[8], row["y"]))
        development_conn.commit()

        # Close database connection
        development_conn.close()

        print("write_data_cluster_success")
        return None

    def get_stay_point(self):
        """
        Extraction of dwell points from trajectory data:
        Note: the data here requires the format of "dataframe"
        :return: None
        """
        # Dictionary classification of trajectory points according to (ID, vehicleid)
        dict_id_car = {}
        for index, row in self.data.iterrows():

            # Filter invalid location data
            if row["av"] == "0":
                continue
            if dict_id_car.get((row["NO"], row["VehicleID"])) is None:
                dict_id_car[(row["NO"], row["VehicleID"])] = []

            # Convert time to timestamp
            ts = time.mktime(time.strptime(row["recvtime"], "%Y-%m-%d %H:%M:%S"))

            # Build a dictionary of (ID, vehicleid)
            dict_id_car[(row["NO"], row["VehicleID"])].append((float(row["lat"]), float(row["lng"]), float(row["veo"]),
                                                               ts, float(row["totaldistance"]), float(row["end_lat"]),
                                                               float(row["end_lng"])))

        # Load all stay points
        lst_stay_point = []

        # (id, VehicleID)-->(Latitude, longitude, speed, time stamp,totaldistance, direction, end latitude, end longitude of receiving the track point)
        for k, v in dict_id_car.items():
            i = 0
            length = len(v)
            flag = False
            for row in v:

                # Record start stay points
                if row[2] <= self.length_of_veo:
                    if flag is False:
                        num = 1
                        sum1 = row[0]
                        sum2 = row[1]
                        start_time = row[3]
                        flag = True
                    else:
                        num = num + 1
                        sum1 = sum1 + row[0]
                        sum2 = sum2 + row[1]

                # Record end stay points
                if row[2] > self.length_of_veo or i == length - 1:
                    if flag is True:
                        flag = False
                        le = row[3] - start_time
                        if le >= self.length_of_time:
                            lst_stay_point.append([k[0], k[1], sum1/num, sum2/num, start_time, row[3], row[4],
                                                   row[5], row[6]])
                i = i + 1

        print(len(lst_stay_point))
        self.sp_data = pd.DataFrame(lst_stay_point).astype(float)

        return None

    def cluster_stay_point(self):
        """
        DBSCAN clustering of stay points
        :return: None
        """

        # DBSCAN clustering
        data = self.sp_data.astype("float")
        y = DBSCAN(eps=200, min_samples=5, metric=get_distance).fit_predict(data)
        self.data_cluster = data
        self.data_cluster["y"] = y
        print(len(y))
        cnt = 0
        for i in y.tolist():
            if i == -1:
                cnt = cnt + 1
        print(cnt)
        print(cnt/len(data))
        m = folium.Map([35.155224, 119.37136299999999], zoom_start=13)
        ma = 0
        for index, row in self.data_cluster.iterrows():
            if row["y"] > -1 and row["y"] % 4 == 0:
                folium.PolyLine([[row[2], row[3]], [row[2], row[3]]], color="red", weight=7).add_to(m)
            if row["y"] > -1 and row["y"] % 4 == 1:
                folium.PolyLine([[row[2], row[3]], [row[2], row[3]]], color="green", weight=7).add_to(m)
            if row["y"] > -1 and row["y"] % 4 == 2:
                folium.PolyLine([[row[2], row[3]], [row[2], row[3]]], color="black", weight=7).add_to(m)
            if row["y"] > -1 and row["y"] % 4 == 3:
                folium.PolyLine([[row[2], row[3]], [row[2], row[3]]], color="blue", weight=7).add_to(m)
            if row["y"] == -1:
                folium.PolyLine([[row[2], row[3]], [row[2], row[3]]], color="orange", weight=7).add_to(m)
            # print(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row["y"])
            if ma < row["y"]:
                ma = row["y"]
        print(ma)
        m.save("final.html")
        webbrowser.open('final.html')

        return None


if __name__ == "__main__":
    """
    Abnormal detection of stay points
    """

    # Read the trajectory data after cleaning
    sp = StayPoint()
    sp.read_data("gps_clearn_up_final_final_auto")

    # Acquisition of dwell point based on trajectory data
    sp.get_stay_point()
    print(len(sp.sp_data))
    sp.write_data_sp("stay_point_yzw_final_30min_auto")
    sp.read_data_sp("stay_point_yzw_final_30min_auto")
    sp.cluster_stay_point()
    sp.write_data_cluster("stay_point_yzw_final_cluster_30min_auto")
    sp.read_data_cluster("stay_point_yzw_final_cluster_30min_auto")
    list_stay_point_null = []
    list_stay_point_cluster = []
    for index, row in sp.data_cluster.iterrows():
        if float(row["label"]) == -1:
            list_stay_point_null.append(row.tolist())
        else:
            list_stay_point_cluster.append(row.tolist())

    print(len(list_stay_point_null))
    print(len(list_stay_point_cluster))

    cnt = 0
    # Three times distance anomaly detection of Gaussian distribution
    lst_label2 = []
    for i in list_stay_point_null:
        dis_min = 1000000000
        label = -1
        for ii in list_stay_point_cluster:
            dis = geodesic((ii[2], ii[3]), (i[2], i[3])).meters
            if dis <= 400 and dis < dis_min:
                dis_min = dis
                label = float(ii[9])
        lst_label2.append(label)

    # Write to database
    development_conn = pymysql.connect(
        host="localhost",  # ip
        database="gps_jingchuang", # database
        user='root',  # user name
        password='yzw161112',  # passord
        port=3344,  # port number
    )
    length = len(list_stay_point_null)
    cnt = 0
    for i in range(length):
        row = list_stay_point_null[i]
        development_conn.cursor().execute("INSERT INTO {0} VALUES('{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}',"
                                          " '{9}','{10}','{11}')".format("stay_point_yzw_final_cluster_30min_auto2",
                                                                  row[0], row[1], row[2], row[3], row[4],
                                                                  row[5], row[6], row[7], row[8], row[9], lst_label2[i]))
        if float(lst_label2[i]) > -1:
            print(row)
            cnt = cnt + 1
        # print(i)
    print(cnt)

    for row in list_stay_point_cluster:
        development_conn.cursor().execute("INSERT INTO {0} VALUES('{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}',"
                                          " '{9}','{10}','{11}')".format("stay_point_yzw_final_cluster_30min_auto2",
                                                                  row[0], row[1], row[2], row[3], row[4],
                                                                  row[5], row[6], row[7], row[8], row[9],
                                                                  row[9]))
    development_conn.commit()

    # Close database connection
    development_conn.close()
    print("write_data_cluster_success")


    """
    (1)Stay point with abnormal location
    """
    development_conn = pymysql.connect(
        host="localhost",  # ip
        database="gps_jingchuang", # database
        user='root',  # user name
        password='yzw161112',  # password
        port=3344,  # port number
    )

    # Read all values of a table in a database
    read_sql = "select * from {0}".format("stay_point_yzw_final_cluster_30min_auto2")
    data_label2 = pd.read_sql(read_sql, development_conn)

    # Close database connection
    development_conn.close()
    print("read_data_success")

    # Record data with label - 1
    list_stay_point_null = []
    list_stay_point_cluster = []
    for index, row in data_label2.iterrows():
        if float(row["label"]) == -1:
            list_stay_point_null.append(row.tolist())
        else:
            list_stay_point_cluster.append(row.tolist())

    # Find location exception
    list_cnt = []
    for i in list_stay_point_null:
        if float(i[10]) > -1:
            continue
        cnt = 0
        for ii in list_stay_point_null:
            dis = geodesic((ii[2], ii[3]), (i[2], i[3])).meters
            if dis <= 400:
                cnt = cnt + 1
        i.append(cnt)

    for i in list_stay_point_null:
        if len(i) == 12 and i[11] == 1:
            print(i)

    # Write to database
    development_conn = pymysql.connect(
        host="localhost",  # ip
        database="gps_jingchuang", # database
        user='root',  # user name
        password='yzw161112',  # password
        port=3344,  # port number
    )
    length = len(list_stay_point_null)
    print(length)
    print(len(list_stay_point_cluster))
    cnt = 0
    for row in list_stay_point_null:
        if len(row) == 12 and row[11] == 1:
            continue
        cnt = cnt + 1
        development_conn.cursor().execute("INSERT INTO {0} VALUES('{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}',"
                                          " '{9}','{10}','{11}')".format("stay_point_yzw_final_cluster_30min_auto2_normal",
                                                                  row[0], row[1], row[2], row[3], row[4],
                                                                  row[5], row[6], row[7], row[8], row[9], row[10]))
    print(cnt)
    for row in list_stay_point_cluster:
        cnt = cnt + 1
        development_conn.cursor().execute("INSERT INTO {0} VALUES('{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}',"
                                          " '{9}','{10}','{11}')".format("stay_point_yzw_final_cluster_30min_auto2_normal",
                                                                  row[0], row[1], row[2], row[3], row[4],
                                                                  row[5], row[6], row[7], row[8], row[9],
                                                                  row[10]))

    print(cnt)
    development_conn.commit()

    # Close database connection
    development_conn.close()
    print("write_data_cluster_success")


    """
    (2)stay point of time anomaly
    """
    development_conn = pymysql.connect(
        host="localhost",  # ip
        database="gps_jingchuang", # database
        user='root',  # user name
        password='yzw161112',  # password
        port=3344,  # port number
    )

    # Read all values of a table in the database
    read_sql = "select * from {0}".format("stay_point_yzw_final_cluster_30min_auto2_normal")
    data_label2_normal = pd.read_sql(read_sql, development_conn)

    # Close database connection
    development_conn.close()
    print("read_data_success")

    dict_label2 = {}
    dict_label1 = []
    for index, row in data_label2_normal.iterrows():
        if float(row[9]) == -1:
            dict_label1.append(row)

        if dict_label2.get(float(row[10])) is None:
            dict_label2[float(row[10])] = []
        dict_label2[float(row[10])].append(row)

    for k, v in dict_label2.items():
        if k == -1:
            continue
        list_time = []
        for row in v:
            list_time.append(float(row[5])-float(row[4]))
        if len(list_time) < 6:
            continue
        # print(list_time)
        percentile = np.percentile(sorted(list_time), (25, 50, 75), interpolation='midpoint')
        print("Quantile:", percentile)

        # Here are five eigenvalues of the box line diagram
        Q1 = percentile[0]
        Q3 = percentile[2]
        IQR = Q3 - Q1
        ulim = Q3 + 1.5 * IQR
        llim = Q1 - 1.5 * IQR
        new_deg = []
        for i in range(len(list_time)):
            if list_time[i] > ulim:
                print(list_time[i])
                # print(v[i])
        print()
        print()

    print(len(dict_label1))
    for row1 in dict_label1:
        list_T = []
        if float(row1[10]) > -1:
            continue
        for row2 in dict_label1:
            mid = (float(row1["start_time"]) - float(row2["start_time"])) % (24*3600)
            if mid <= 3600:
                list_T.append(float(row2[5])-float(row2[4]))
        # print(float(row1[5])-float(row1[4]))
        # print(list_T)
        if len(list_T) < 6:
            continue
        percentile = np.percentile(sorted(list_T), (25, 50, 75), interpolation='midpoint')

        # Here are five eigenvalues of the box line diagram
        Q1 = percentile[0]
        Q3 = percentile[2]
        IQR = Q3 - Q1
        ulim = Q3 + 1.5 * IQR
        for i in list_T:
            if i > ulim and i == float(row1[5])-float(row1[4]):
                print(i)
                print()
                print()
