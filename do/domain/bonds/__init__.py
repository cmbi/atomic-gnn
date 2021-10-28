import os
import csv
import logging


from do.models.bond import BondDataObject


bonds_directory_path = os.path.dirname(os.path.abspath(__file__))


_log = logging.getLogger(__name__)


def get_bond_data(name):
    bond_orders = {"doub": 2, "sing": 1}

    bond_data = []
    path = os.path.join(bonds_directory_path, "%s.csv" % name)
    if not os.path.isfile(path):
        _log.warning("no bond data for {}".format(name))
        return []

    with open(path, 'rt') as f:
        r = csv.reader(f, delimiter=',')

        header = next(r)
        variable_indices = {name: index for index, name in enumerate(header)}

        for row in r:
            atom1_name = row[variable_indices["First Atom"]]
            atom2_name = row[variable_indices["Second Atom"]]
            if atom1_name == atom2_name:
                raise ValueError("{}: bond of {} with itself".format(path, atom1_name))

            bond_order_type = row[variable_indices["Bond Order Type"]]
            bond_order = bond_orders[bond_order_type]

            bond_length = float(row[variable_indices["Bond Length"]])

            bond_data.append(BondDataObject(atom1_name, atom2_name, bond_order, bond_length))

    return bond_data



