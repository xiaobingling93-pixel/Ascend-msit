import psutil

SYS_TS = psutil.boot_time()


def convert_syscnt_to_ts(cnt, start_cnt, cpu_frequency):
    return (SYS_TS + ((cnt - start_cnt) / cpu_frequency)) * 1000 * 1000