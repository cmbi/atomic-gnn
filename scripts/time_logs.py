import re
import sys


PATTERN1 = re.compile(r"took (\d+\.\d+) seconds to ([a-z ]+)")
PATTERN2 = re.compile(r"([A-Za-z0-9_]+) took (\d+\.\d+) seconds")

job_counts = {}
time_per_job = {}

with open(sys.argv[1]) as logfile:
    for line in logfile:
        i = line.rfind(':')
        if i < 0:
            continue

        statement = line[i + 1:].strip()
        m1 = PATTERN1.match(statement)
        m2 = PATTERN2.match(statement)
        if m1:
            time_s = m1.group(1)
            job = m1.group(2)

        elif m2:
            job = m2.group(1)
            time_s = m2.group(2)
        else:
            continue

        if job not in time_per_job:
            time_per_job[job] = 0.0
            job_counts[job] = 0

        job_counts[job] += 1
        time_per_job[job] += float(time_s)

total_seconds = sum(time_per_job.values())

for job, seconds in time_per_job.items():
    print(job, ':' ,"%.1f" % (100 * seconds / total_seconds), '%', "(n=%d)" % job_counts[job])
