import psutil

SYS_TS = psutil.boot_time()
US_PER_SECOND = 1000 * 1000


def convert_syscnt_to_ts(cnt, start_cnt, cpu_frequency):
    return (SYS_TS + ((cnt - start_cnt) / cpu_frequency)) * US_PER_SECOND