#!/usr/bin/env python

import sys
import h5py


def read(data_paths):
    "read the contents of the data paths and count the class values"

    class_value_counts = {}
    for data_path in data_paths:
        with h5py.File(data_path, 'r') as hdf5_file:
            entries_count = len(hdf5_file.keys())
            for entry_name in hdf5_file.keys():
                class_data = hdf5_file['{}/score/bin_class'.format(entry_name)][()]
                if class_data.shape != (1,):
                    raise ValueError("{}: got {} for class".format(entry_name, class_data))

                class_value = class_data.item(0)
                if class_value not in class_value_counts:
                    class_value_counts[class_value] = 0
                class_value_counts[class_value] += 1

    return class_value_counts


if __name__ == "__main__":
    counts = read(sys.argv[1:])

    for class_, count in counts.items():
        print("class", class_, ":", count, "entries")
