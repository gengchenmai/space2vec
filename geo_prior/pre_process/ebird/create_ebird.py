"""
Loads in the big eBird checklist dataset for a given year and stores the indicies
of species that have more than one observation for each checklist.
"""

import json
import calendar
import numpy as np

year = 2016
file_name = '/media/macaodha/Data/ebird/ERD2016SS/'+str(year)+'/checklists.csv'
op_file_name = str(year) + '_ebird_raw.json'
print('Processing year ' + str(year))

num_days = 366.0 if calendar.isleap(year) else 365.0
num_days -= 1

lon_lat_tm = []
check_lists = []
observer_ids = []
with open(file_name) as input_file:
    for ii, line in enumerate(input_file):
        if ii == 0:
            species = line.replace('\r\n', '').split(',')[19:]

        elif ii > 0:
            # X indicates "present-without-count"
            line_c = line.replace(',X', ',1').replace(',?', ',0')
            ll = line_c.split(',')

            if ll[18] == '1':  # use PRIMARY_CHECKLIST_FLAG to remove duplicates from groups
                date_continuous = (int(ll[6])-1) / num_days
                # lon, lat, day
                loc = (round(float(ll[3]),5), round(float(ll[2]),5), round(date_continuous, 5))
                lon_lat_tm.append(loc)

                observer_ids.append(ll[15])

                llint = np.array(ll[19:]).astype(np.int)
                check_lists.append(np.where(llint>1)[0].tolist())

        if ii%1000==0:
            print ii

op = {}
op['lon_lat_tm'] = lon_lat_tm
op['check_lists'] = check_lists
op['observer_ids'] = observer_ids
op['species'] = species

print('saving data to: ' + op_file_name)
with open(op_file_name, 'w') as da:
    json.dump(op, da, indent=2)
