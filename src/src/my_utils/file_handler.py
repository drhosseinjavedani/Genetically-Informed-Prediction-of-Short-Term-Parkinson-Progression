
# Convert xlsx file to csv
###################################################
###################################################
###################################################
###################################################

import csv
import xlrd


def csv_from_excel(xls_file_path, csv_file_path):
    """
    Convert xlsx file to csv
    """
    # open workbook by sheet index,
    # optional - sheet_by_index()
    sheet = xlrd.open_workbook(xls_file_path).sheet_by_index(0)

    # writer object is created
    col = csv.writer(open(csv_file_path, "w", newline=""))

    # writing the data into csv file
    for row in range(sheet.nrows):
        # row by row, write
        # operation is perform
        col.writerow(sheet.row_values(row))
    csv_file_path
