#!/usr/bin/env python

# This script downloads bond data from PDBeChem

import csv
from urllib import request
import sys
import logging
import xml.etree.ElementTree as ET


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
_log = logging.getLogger(__name__)



def download(three_letter_code, path):
    with request.urlopen("https://www.ebi.ac.uk/pdbe-srv/pdbechem/bond/list/%s" % three_letter_code.upper()) as f:
        html = f.read().decode("ascii")

    start_index = html.find("<table id=\"resultsTable\" ")
    if start_index == -1:
        raise ValueError("Cannot find results table")

    end_index = html.find("</table>", start_index)
    if end_index == -1:
        raise ValueError("Cannot find end of table")

    end_index += 8

    table_html = html[start_index: end_index]

    root = ET.fromstring(table_html)

    table = root.find(".")

    thead, tbody = table

    with open(path, 'wt') as f:
        w = csv.writer(f, delimiter=',')

        header = thead.find("tr")
        header_names = [th.text for th in header.findall("th")]
        w.writerow(header_names)

        for row in tbody.findall("tr"):

            values = []
            for td in row.findall("td"):
                a = td.find("a")
                if a is None:
                    values.append(td.text)
                else:
                    values.append(a.text)

            w.writerow(values)


if __name__ == "__main__":

    for amino_acid_code in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
                            'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'SEC', 'PYL']:

        table = download(amino_acid_code, "%s.csv" % amino_acid_code)
