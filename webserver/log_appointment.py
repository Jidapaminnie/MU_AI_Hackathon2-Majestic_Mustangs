import pandas as pd
from datetime import datetime, timedelta, timezone
import csv
import os
import uuid

log_path = os.path.join("data", "log_appointment.csv")
COLS = ['uuid', 'doctorID', 'patientName', 'datetimeStart', 'datetimeEnd', 'duration_min']
datetimeFormat = "day of the week: %a, day: %d, month: %b, year: %Y, time: %H:%M:%S"

def write_row_csv(path, new_row):
    with open(path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        if isinstance(new_row, dict):
            csvwriter.writerow(new_row.values())
        else: csvwriter.writerow(new_row)


def create_appoint_log(num_week=5, log_path=log_path, delete_existing=True):
    # days = ['Sun', 'Mon', 'Tues', 'Wed', 'Thrus', 'Fri', 'Sat']
    # COLS = [] # ['slotName', 'slotStart', 'slotFin']
    # for week in range(1, num_week + 1):
    #     COLS.extend(["week_" + str(week) + "_" + day for day in days])
    # slots = []
    if delete_existing and os.path.isfile(log_path):
        os.remove(log_path)
    write_row_csv(log_path, COLS)

def check_time_validity(datetimeStart, duration_hr, datetimeEnd):
    
    if (datetimeEnd is None) and (duration_hr is not None):
        datetimeEnd = datetimeStart + timedelta(hours=duration_hr)
    elif (datetimeEnd is not None) :
        duration_hr = (datetimeEnd - datetimeStart).total_seconds()/3600
    # elif (datetimeEnd is not None) and (duration_hr is not None):
    assert (datetimeEnd - datetimeStart).total_seconds()/3600 == duration_hr, "given dateteime data is invalid"
    return duration_hr, datetimeEnd


def add_appointment(doctorID: int, patientName: str, datetimeStart: datetime, # required
                    duration_min: int = 1, datetimeEnd: datetime=None):
    ''' 
    Must specify (datetimeStart & duration_min) OR (datetimeStart & datetimeEnd). If both are true, duration_hr is changed based on datetimeEnd
    '''
    

    duration_hr, datetimeEnd = check_time_validity(datetimeStart, duration_hr, datetimeEnd)

    status, overlap = check_appointment_overlap(doctorID, datetimeStart, duration_min, datetimeEnd)
    assert status, overlap
    app_dict = {}
    app_dict['appointment_uuid'] = uuid.uuid4()
    app_dict['doctorID'] = doctorID
    app_dict['patientName'] = patientName
    app_dict['datetimeStart'] = datetimeStart.isoformat()# .strftime(datetimeFormat)
    app_dict['datetimeEnd'] = datetimeEnd.isoformat()# .strftime(datetimeFormat)
    app_dict['duration_min'] = duration_min
    write_row_csv(log_path, app_dict)
    return app_dict

def get_appointment_by_doctor(doctorID:int, log_path=log_path):
    df = pd.read_csv(log_path)
    df = df[df["doctorID"] == doctorID]
    return df.to_dict('records')

def get_appointment_by_uuid(uuid, log_path=log_path):
    df = pd.read_csv(log_path)
    df = df[df["uuid"] == uuid]
    return df

def get_appointment(col:str, val, log_path=log_path):
    df = pd.read_csv(log_path)
    df = df[df[col] == val]
    return df.to_dict('records')

def check_appointment_overlap(doctorID, datetimeStart: datetime,
                    duration_hr: int = 1, datetimeEnd: datetime=None):
    '''
    return (status in Boolean, list of overlapping appointments in Series)
    '''
    duration_hr, datetimeEnd = check_time_validity(datetimeStart, duration_hr, datetimeEnd)
    df = pd.read_csv(log_path)
    df = df[df["doctorID"] == doctorID]
    l = []
    for i, log in df.iterrows():
        # if (datetimeStart < datetime.strptime(log['datetimeEnd'], datetimeFormat)) and (datetime.strptime(log['datetimeStart'], datetimeFormat) < datetimeEnd):
        if (datetimeStart < datetime.fromisoformat(log['datetimeEnd'])) and (datetime.fromisoformat(log['datetimeStart']) < datetimeEnd) :
            l.append(log)
    if len(l) > 0:
        return False, l # overlap
    else:
        return True, l # no overlap



# create_appoint_log()
# add_appointment(0, "patient A", datetime(2024, 12, 12, 14, 30))
# add_appointment(0, "patient B", datetime(2024, 12, 12, 15, 30))
# add_appointment(0, "patient D", datetime(2024, 12, 12, 16, 30), 0.5)
# add_appointment(0, "patient C", datetime(2024, 12, 13, 12, 00), 2, datetime(2024, 12, 13, 17, 30))
# check_appointment_overlap(0, datetime(2024, 12, 12, 15, 00))

# print(get_appointment_by_doctor(0))