import webbrowser
import datetime
import folium
import pymysql
import time
import gc
import pandas as pd
import copy
from sklearn.cluster import DBSCAN, OPTICS, MeanShift
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import copy
import requests


"""
1, Read the required database
"""
# Connect to database
development_conn = pymysql.connect(
    host="localhost",  # IP address
    database="gps_jingchuang",  # Database table name
    user='root',  # User name
    password='yzw161112',  # Password
    port=3344,  # Port number
)

# Read data of database
read_sql = "select VehicleID, recvtime, lng, lat, veo, direct, av, totaldistance  from {0} ".format("ods_sqlserver_gps_use")
data_GPS = pd.read_sql(read_sql, development_conn)

# Read data of database
read_sql = "select * from {0} ".format("1_car_no_yzw_08")
car_to_NO = pd.read_sql(read_sql, development_conn)

# Building the dictionary of "car" and "ID"
dict_NO_car = {}
dict_car_NO = {}
for index, row in car_to_NO.iterrows():
    dict_car_NO[row["NO"]] = row["ID"]
    dict_NO_car[row["ID"]] = row["NO"]

# Read data of database
read_sql = "select * from {0}".format("end_point_info_auto")
end_info = pd.read_sql(read_sql, development_conn)

# Building the dictionary of "location_id" and "(latitude,longitude)"
dict_end_lat_lng = {}
for index, row in end_info.iterrows():
    dict_end_lat_lng[row["location_id"]] = (row["latitude"], row["longitude"])

# Read data of database
read_sql = "select * from {0}".format("yzw_start_end_time_all_auto")
start_end_info = pd.read_sql(read_sql, development_conn)

# Close database connection
development_conn.close()


"""
2,Extraction of trajectory data
"""
# Build a dictionary
data_GPS_car_dict = {}
for index, row in data_GPS.iterrows():
    if data_GPS_car_dict.get(row[0]) is None:
        data_GPS_car_dict[row[0]] = []
    data_GPS_car_dict[row[0]].append([row[1], row[2], row[3], row[4], row[5], row[6], row[7]])

# Recovery memory
del data_GPS
gc.collect()

# Remove duplicate orders
cnt = 0
start_end_info = start_end_info.drop_duplicates()

# Processing of trajectory data
for k, v in dict_NO_car.items():
    lst_start_end = []

    # Select order based on "car"
    for index, row in start_end_info.iterrows():
        if row[0] == v:
            lst_start_end.append([row[0], row[1], row[2], row[3], str(row[3])[:10]])

    # Multiple loading information of a vehicle in the same day
    dict_index_start_end_time = {}
    for row in lst_start_end:
        if dict_index_start_end_time.get(row[4]) is None:
            dict_index_start_end_time[row[4]] = []
        dict_index_start_end_time[row[4]].append([row[0], row[1], row[2], row[3]])

    # Record format（Date, vehicle）->(end1，end2，...)
    dict_end_mul = {}

    # Record format（Date, vehicle）->(Reference time, reference start point, reference end point)
    dict_time_mul = []
    for kk, vv in dict_index_start_end_time.items():
        if dict_end_mul.get((kk, vv[0][0])) is None:
            dict_end_mul[(kk, vv[0][0])] = []
        tmp_t = "1918-09-25 18:31:55"
        for i in vv:
            dict_end_mul[(kk, vv[0][0])].append(i[2])
            if tmp_t < str(i[3]):
                tmp_t = str(i[3])
        dict_time_mul.append([kk, vv[0][0], tmp_t])

    # Sort by time
    dict_time_mul = sorted(dict_time_mul, key=(lambda x: x[2]))
    length = len(dict_time_mul)
    print(length)

    for i in range(0, length):
        if i == 0:
            dict_time_mul[i].extend(["1918-09-25 18:31:55", dict_time_mul[i+1][2]])
        elif i == length - 1:
            dict_time_mul[i].extend([dict_time_mul[i-1][2], "2918-09-25 18:31:55"])
        else:
            dict_time_mul[i].extend([dict_time_mul[i-1][2], dict_time_mul[i + 1][2]])

    # print dict_time_ul
    for row in dict_time_mul:
        print(row)
        # print(dict_end_mul[(row[0], row[1])])
    print()
    print()

    # record start and end
    num_start = []
    num_end = []
    data_GPS_sort = sorted(data_GPS_car_dict[k], key=(lambda x: x[0]))
    length = len(data_GPS_sort)
    print(length)
    trans = 0

    # Data cleaning
    for row in dict_time_mul:
        print(row)
        print(row[2])
        end_point_list = []
        for ele in dict_end_mul[(row[0], row[1])]:

            # End coordinates not found
            if dict_end_lat_lng.get(ele) is None:
                continue
            end_point_list.append(dict_end_lat_lng[ele])
            # print(dict_end_lat_lng[ele])
        print(end_point_list)

        # End coordinate does not exist
        if len(end_point_list) == 0:
            print("continue")
            continue

        # Marking of various situations
        flag = False
        flag_start = False
        flag_end = False

        # Record start point
        start = -1

        # Prepare for backtracking
        end = -1

        # Is it backtracking
        back = False

        # Judge whether it is out of line
        jump = False

        # The goal is to traverse once to get the cleaning result
        for i in range(trans, length):

            # End cycle
            if str(row[4]) <= str(data_GPS_sort[i][0]):
                trans = i
                break

            # Find a reference point in time
            if jump is False and flag is False and str(data_GPS_sort[i][0]) >= str(row[2]):
                print(data_GPS_sort[i][0])
                print(row[2])

                if str(data_GPS_sort[i][0]) >= str(row[4]):
                    print("break")
                    break

                # Record the reference time point
                flag = True
                dis_tmp = geodesic((data_GPS_sort[i][2], data_GPS_sort[i][1]), (35.157347, 119.337247))
                print(dis_tmp)
                print(i)

                # Mark no need to backtrack
                if dis_tmp < 2:
                    start = i

                # There may be a need for backtracking
                else:
                    end = i + 1

            # Find the first trajectory point 2km from the starting point
            if jump is False and flag is True:
                if geodesic((data_GPS_sort[i][2], data_GPS_sort[i][1]), (35.157347, 119.337247)) < 2:
                    start = i

            # Find the first trajectory point 10km from the end point
            if jump is False:
                for ll in end_point_list:
                    if flag is True and geodesic((data_GPS_sort[i][2], data_GPS_sort[i][1]), ll) < 10:
                        print(geodesic((data_GPS_sort[i][2], data_GPS_sort[i][1]), ll))
                        print(data_GPS_sort[i][0])
                        flag_end = True
                        num_end.append((i, ll))
                        trans = i
                        print(i)
                        print(ll)
                        back = True
                        jump = True
                        break

            # backtracking
            if back is True:
                if start == -1:
                    for ii in reversed(range(0, end)):
                        if str(row[3]) >= str(data_GPS_sort[i][0]):
                            break
                        if geodesic((data_GPS_sort[ii][2], data_GPS_sort[ii][1]), (35.157347, 119.337247)) < 2:
                            print(geodesic((data_GPS_sort[ii][2], data_GPS_sort[ii][1]), (35.157347, 119.337247)))
                            print("backtracking")
                            num_start.append(ii)
                            flag_start = True
                            print(ii)
                            break

                # Once found, record the starting point
                else:
                    print("NO backtracking")
                    print(start)
                    print(geodesic((data_GPS_sort[start][2], data_GPS_sort[start][1]), (35.157347, 119.337247)))
                    num_start.append(start)
                    flag_start = True
                print()
                print()

                # Delete redundant records
                if flag_start is False and flag_end is True:
                    num_end.pop()
                if flag_end is False and flag_start is True:
                    num_start.pop()
                back = False

    print(len(lst_start_end))
    # print(num_start)
    # print(num_end)
    print(len(num_start))
    print(len(num_end))
    cnt = cnt + len(num_end)


    """
    3,Write the extracted track data to the database
    """
    # Connect to database
    development_conn = pymysql.connect(
        host="localhost",  # ip
        database="gps_jingchuang", # database name
        user='root',  # user name
        password='yzw161112',  # password
        port=3344,  # port number
    )

    for i in range(len(num_start)):
        if num_start[i] < num_end[i - 1][0] and i > 0:
            print("False")
            cnt = cnt - 1
            continue

        for ii in range(num_start[i], num_end[i][0]+1):
            yy = data_GPS_sort[ii]
            # print(num_start[i], num_end[i][0])
            # print(i, k, yy[0], yy[1], yy[2], yy[3], yy[4], yy[5], yy[6], num_end[i][1][1], num_end[i][1][0])
            development_conn.cursor().execute("INSERT INTO GPS_clearn_up_final_final_auto VALUES ('{0}', '{1}', '{2}',"
                                              " '{3}','{4}', '{5}','{6}','{7}','{8}','{9}','{10}')".
                                              format(i, k, yy[0], yy[1], yy[2], yy[3], yy[4], yy[5], yy[6],
                                                     num_end[i][1][1], num_end[i][1][0]))
    development_conn.commit()

    # Close database connection
    development_conn.close()
print(cnt)




