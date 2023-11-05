import random
import numpy as np
import pandas
from datetime import datetime
from typing import Optional
import os
random.seed(42)
np.random.seed(42)

TARGET_DATA_POINTS = (1000000)

def generate_data(TARGET_DATA_POINTS= TARGET_DATA_POINTS) -> pandas.DataFrame:
    # generate[] uid
    uids = list(range(1, TARGET_DATA_POINTS +1))
    # pid
    p_variant = ['L'] * 500000 + ['M'] * 300000 + ['H'] * 200000
    p_serial = np.random.randint(1000, 9999, size=TARGET_DATA_POINTS)
    p_id = [variant + str(serial) for variant, serial in zip(p_variant, p_serial)]

    air_temp = np.random.normal(300, 2, size=TARGET_DATA_POINTS)
    p_temp = air_temp + np.random.normal(0,1, size=TARGET_DATA_POINTS)


    power = 2860 

    rot_speed = np.random.normal(power, 50, size=TARGET_DATA_POINTS)

    torque = np.abs(np.random.normal(40, 10, size=TARGET_DATA_POINTS))

    t_wear = {'L': 2, 'M':3, 'H':5}
    t_wear_values = [t_wear[variant] for variant in p_variant]

    prob_failure = 0.01

    m_failure_label = [1 if random.random() < prob_failure else 0 for _ in range(TARGET_DATA_POINTS)]

    data = {"uids": uids,
            "p_id": p_id,
            "air_temp": air_temp,
            "process_temp": p_temp,
            "rot_speed": rot_speed,
            "torque": torque,
            "tool_wear": t_wear_values,
            "failure": m_failure_label}

    df = pandas.DataFrame(data)

    return df    
def df_to_csv(df:pandas.DataFrame) -> os.PathLike:
    df.to_csv(f"output/realtime_data.csv")
    return os.path.join(f"output/realtime_data.csv")

# TODO: implement method to integrate with local_db
def to_sqlite3() -> None:
    pass


if __name__ == "__main__":
    df_to_csv(generate_data())
