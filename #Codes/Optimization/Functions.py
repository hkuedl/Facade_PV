import pandas as pd
import numpy as np

def TOU_period(cc):
    price_file = pd.read_excel('Realtime Price.xlsx',sheet_name = 'Sheet1')

    row_index = price_file[price_file.eq(cc)].any(axis=1).idxmax()

    cc_province = price_file.iloc[row_index,0]
    CarbonF = price_file.iloc[row_index,-1]

    TOU_time = np.zeros((12,24))

    numbers = list(range(1,13))
    if cc_province == '广东':
        for i in [7,8,9]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,2,1,3,3,2,1,1,2,2,3,3,3,3,3]
        for i in [num for num in numbers if num not in {7,8,9}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,2,2,3,3,2,2,2,2,2,3,3,3,3,3]
    if cc_province == '江苏':
        for i in numbers:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,2,2,2,3,3,3,3,3,3,2,2,2,2,2,3,3]
    if cc_province == '山西':
        for i in [1,7,8,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,2,2,2,4,4,3,3,3,3,2,1,1,2,2,2,3]
        for i in [num for num in numbers if num not in {1,7,8,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,2,2,2,4,4,3,3,3,3,2,2,2,2,2,2,3]
    if cc_province == '浙江':
        for i in [1,7,8,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,2,1,1,4,4,3,3,1,1,2,2,2,2,2,2,3]
        for i in [num for num in numbers if num not in {1,7,8,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,2,2,2,4,4,2,2,2,2,3,3,3,3,3,3,3]
    if cc_province == '重庆':
        for i in [1,7,8,9,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,3,2,1,1,2,2,2,3,3,3,2,2,3,3]
        for i in [num for num in numbers if num not in {1,7,8,9,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,3,2,2,2,2,2,2,3,3,3,2,2,3,3]
    if cc_province == '天津':
        for i in [7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,2,2,1,3,3,3,3,1,2,2,2,2,3,3,4]
        for i in [num for num in numbers if num not in {7,8}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,2,2,2,3,3,3,3,2,2,2,2,2,3,3,4]
    if cc_province == '湖南':
        for i in [1,7,8,9,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,3,2,2,2,3,3,3,3,1,1,1,1,2,4]
        for i in [num for num in numbers if num not in {1,7,8,9,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,3,2,2,2,3,3,3,3,2,2,2,2,2,4]
    if cc_province == '海南':
        for i in [5,6,7]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,3,2,2,2,2,1,1,3,4]
        for i in [num for num in numbers if num not in {5,6,7}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,3,2,2,2,2,2,2,3,4]
    if cc_province == '陕西':
        for i in [7,8]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,3,3,3,3,3,4,4,4,4,2,2,2,2,1,1,2,2,3]
        for i in [1,12]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,3,3,3,3,3,4,4,4,4,2,2,2,1,1,2,2,2,3]
        for i in [num for num in numbers if num not in {1,7,8,12}]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,3,3,3,3,3,4,4,4,4,2,2,2,2,2,2,2,2,3]
    if cc_province == '黑龙江':
        for i in [1,7,8,9,11,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,2,3,3,2,2,2,3,3,3,3,2,1,1,2,3,3,3,4]
        for i in [num for num in numbers if num not in {1,7,8,9,11,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,2,3,3,2,2,2,3,3,3,3,2,2,2,2,3,3,3,4]
    if cc_province == '辽宁':
        for i in [1,7,8,11,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,3,3,3,2,2,2,3,4,3,3,3,2,1,1,2,2,3,4,4]
        for i in [num for num in numbers if num not in {1,7,8,11,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,3,3,3,2,2,2,3,4,3,3,3,2,2,2,2,2,3,4,4]
    if cc_province == '吉林':
        for i in [1,2,7,8,11,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,2,2,2,3,3,3,3,1,1,2,2,2,3,3,4]
        for i in [num for num in numbers if num not in {1,2,7,8,11,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,2,2,2,3,3,3,3,2,2,2,2,2,3,3,4]
    if cc_province == '四川':
        for i in [7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,1,1,2,2,2,2,3,3,4]
        for i in [1,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,2,2,2,2,1,1,3,3,4]
        for i in [num for num in numbers if num not in {1,7,8,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,2,2,2,2,2,2,3,3,4]
    if cc_province == '冀北':
        for i in [6,7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,3,3,3,3,2,1,1,1,2,2,3,4]
        for i in [1,11,12]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,4,3,2,2,3,3,4,4,3,3,2,1,1,2,2,2,3,3]
        for i in [num for num in numbers if num not in {1,6,7,8,11,12}]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,4,3,2,2,3,3,4,4,3,3,2,2,2,2,2,2,3,3]
    if cc_province == '福建':
        for i in [7,8,9]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,2,1,3,3,3,2,2,1,2,2,3,2,3,3]
        for i in [num for num in numbers if num not in {7,8,9}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,2,2,3,3,3,2,2,2,2,2,3,2,3,3]
    if cc_province == '北京':
        for i in [7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,1,1,3,3,3,1,2,2,2,2,2,3,4]
        for i in [1,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,2,3,3,3,3,2,1,1,1,2,3,4]
        for i in [num for num in numbers if num not in {1,7,8,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,2,2,2,3,3,3,3,2,2,2,2,2,3,4]
    if cc_province == '青海':
        for i in [1,2,3,10,11,12]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,1,4,4,4,4,4,4,4,4,2,2,1,1,2,2,3]
        for i in [num for num in numbers if num not in {1,2,3,10,11,12}]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,2,4,4,4,4,4,4,4,4,2,2,2,2,2,2,3]
    if cc_province == '云南':
        for i in [1,3,4,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,2,2,1,3,3,3,3,3,2,1,2,2,2,3,4]
        for i in [num for num in numbers if num not in {1,3,4,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,2,2,2,3,3,3,3,3,2,2,2,2,2,3,4]
    if cc_province == '蒙东':
        for i in numbers:
            TOU_time[i-1,:] = [4,4,4,4,4,3,2,2,2,3,3,4,4,4,3,3,3,2,2,2,2,2,3,3]
    if cc_province == '蒙西':
        for i in [6,7,8]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,2,2,3,3,3,4,4,4,4,4,3,3,2,2,2,2,3,3]
        for i in [num for num in numbers if num not in {6,7,8}]:
            TOU_time[i-1,:] = [4,4,4,4,3,3,2,2,3,3,3,4,4,4,4,4,3,3,2,2,2,2,3,3]
    if cc_province == '新疆':
        for i in [7]:
            TOU_time[i-1,:] = [3,3,3,3,4,4,4,4,2,2,2,3,3,4,5,5,4,3,3,2,2,1,1,2]
        for i in [1,11,12]:
            TOU_time[i-1,:] = [3,3,3,3,4,4,4,4,2,2,2,3,3,4,4,4,4,3,3,1,1,2,2,2]
        for i in [5,6,8]:
            TOU_time[i-1,:] = [3,3,3,3,4,4,4,4,2,2,2,3,3,4,5,5,4,3,3,2,2,2,2,2]
        for i in [num for num in numbers if num not in {1,5,6,7,8,11,12}]:
            TOU_time[i-1,:] = [3,3,3,3,4,4,4,4,2,2,2,3,3,4,4,4,4,3,3,2,2,2,2,2]
    if cc_province == '山东':
        for i in [1,2,12]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,3,3,3,4,5,5,5,4,3,1,1,1,2,2,3,3,3]
        for i in [3,4,5]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,3,3,3,4,5,5,5,4,3,3,1,1,1,2,2,3,3]
        for i in [6,7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,2,1,1,1,1,1,3,3]
        for i in [9,10,11]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,3,3,3,4,5,5,5,4,3,2,1,1,2,2,3,3,3]
    if cc_province == '贵州':
        for i in numbers:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,2,2,2,3,3,3,3,2,2,2,2,2,3,3]
    if cc_province == '广西':
        for i in numbers:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,3,2,2,3,3,3,3,2,2,2,2,2,2,4]
    if cc_province == '宁夏':
        for i in [7,8]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,2,4,4,4,4,4,4,4,4,2,2,1,1,2,2,3]
        for i in [12]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,2,4,4,4,4,4,4,4,4,2,1,1,2,2,2,3]
        for i in [4,5,9]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,2,4,4,4,5,5,4,4,4,2,2,2,2,2,2,3]
        for i in [num for num in numbers if num not in {4,5,7,8,9,12}]:
            TOU_time[i-1,:] = [3,3,3,3,3,3,3,2,2,4,4,4,4,4,4,4,4,2,2,2,2,2,2,3]
    if cc_province == '甘肃':
        for i in numbers:
            TOU_time[i-1,:] = [3,3,3,3,3,3,2,2,3,3,4,4,4,4,4,4,3,3,2,2,2,2,2,3]
    if cc_province == '上海':
        for i in numbers:
            TOU_time[i-1,:] = [4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4]
    if cc_province == '安徽':
        for i in [7,8,9]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,2,2,2,2,1,1,2,2]
        for i in [1,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,2,2,2,2,1,1,2,2,4]
        for i in [num for num in numbers if num not in {1,7,8,9,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,2,2,2,3,3,3,3,3,2,2,2,2,2,3,3,4]
    if cc_province == '湖北':
        for i in [7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,3,4,4,3,3,2,2,2,2,1,1,2,2]
        for i in [num for num in numbers if num not in {7,8}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,3,4,4,3,3,2,2,1,1,2,2,2,2]
    if cc_province == '江西':
        for i in [1,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,2,2,2,3,3,3,3,3,3,1,1,2,3,3,3]
        for i in [7,8,9]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,1,1,3]
        for i in [num for num in numbers if num not in {1,7,8,9,12}]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,3,3]
    if cc_province == '河南':
        for i in [1,2,12]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,2,1,1,2,2,2,2,2]
        for i in [3,4,5,9,10,11]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,3,3,3,3,3,4,4,4,3,3,2,2,2,2,2,2,2,2]
        for i in [6,7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,2,2,2,2,1,1,1,2]
    if cc_province == '河北':
        for i in [6,7,8]:
            TOU_time[i-1,:] = [4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,2,2,2,2,1,1,1,2,3]
        for i in [1,2,12]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,3,3,3,3,3,3,4,4,4,3,2,1,1,2,2,2,2,2]
        for i in [num for num in numbers if num not in {1,2,6,7,8,12}]:
            TOU_time[i-1,:] = [3,4,4,4,4,4,3,3,3,3,3,3,4,4,4,3,2,2,2,2,2,2,2,2]
    
    TOU_price_indu = np.zeros((12,24))
    TOU_price_resi = np.zeros((12,24))
    
    TOU_price_coal = price_file.iloc[row_index,2]
    for mm in range(1,13):
        mm0 = 11 if mm == 12 else mm
        TOU = pd.read_excel('Realtime Price.xlsx',sheet_name = str(mm0))
        if cc_province in TOU['地区'].values:
            row_index = TOU[TOU['地区'] == cc_province].index[0]
            for ii in range(24):
                tou_i = 4 if int(TOU_time[mm-1,ii]) == 5 else int(TOU_time[mm-1,ii])
                TOU_price_indu[mm-1,ii] = TOU.iloc[row_index,tou_i]
            TOU_price_resi[mm-1,:] = 0.3*TOU.iloc[row_index,3] + 0.7*TOU.iloc[row_index,4]

    return TOU_time,TOU_price_indu,TOU_price_resi,TOU_price_coal,CarbonF

def reorder():
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    def SSR(power,load):
        monday = [31,28,31,30,31,30,31,31,30,31,30,31]
        hour = np.zeros((365,24))
        day = np.zeros((365,1))
        month = np.zeros((12,1))
        for i in range(365):
            for j in range(24):
                hour[i,j] = min([power[i*24+j], load[i*24+j]])/load[i*24+j]
            power_s,load_s = sum(power[i*24:(i+1)*24]),sum(load[i*24:(i+1)*24])
            day[i,0] = min([power_s,load_s])/load_s
        for m in range(12):
            power_m = sum(power[sum(monday[:m])*24:sum(monday[:m+1])*24])
            load_m = sum(load[sum(monday[:m])*24:sum(monday[:m+1])*24])     
            month[m,0] = min(power_m,load_m)/load_m
        return hour, day, month
    case = 1
    D_load_pd = pd.read_hdf('total_load_cities_df.hdf')
    D_load = np.array(D_load_pd)/1e3
    D_roof = np.load('Power_roof_'+str(case)+'.npy')
    D_fa_id = np.load('Power_facade_ideal_'+str(case)+'.npy')
    City_hour = np.zeros((102,4))
    for i in range(102):
        monday = [31,28,31,30,31,30,31,31,30,31,30,31]
        [hour_r,day_r,month_r] = SSR(D_roof[i,:],D_load[i,:])
        [hour_rf,day_rf,month_rf] = SSR(D_fa_id[i,:]+D_roof[i,:],D_load[i,:])
        list_sea = [[i for i in range(sum(monday[:2]),sum(monday[:5]))],[i for i in range(sum(monday[:5]),sum(monday[:8]))],\
                        [i for i in range(sum(monday[:8]),sum(monday[:11]))],[i for i in range(sum(monday[:2]))] + [i for i in range(sum(monday[:11]),sum(monday))]]
        for j in range(4):
            City_hour[i,j] = (np.sum(hour_rf[list_sea[j],:] >= 1) - np.sum(hour_r[list_sea[j],:] >= 1))/len(list_sea[j])

    building_volume_df = pd.read_excel('City_statistic.xlsx', index_col=0)  
    building_volume_df = building_volume_df.drop(columns='Total(km3)')
    building_volume_df.columns = building_volume_df.columns.astype(str)
    total_building_volume = building_volume_df.sum(axis=1)
    key_information = pd.read_excel('City_statistic.xlsx', sheet_name=2, index_col=0)
    total_building_area = key_information['Area_roof-0(km2)']
    average_building_height = total_building_volume / total_building_area * 1000
    facade_roof_ratio = key_information['Area_facade-0(km2)'] / key_information['Area_roof-0(km2)']

    ss_hour_increase_df = pd.DataFrame(City_hour)
    ss_hour_increase_df.index = building_volume_df.index

    cities_hku_dest = pd.read_hdf('cities_hku_dest.hdf', key='cities_hku_dest')
    location_cities = cities_hku_dest[['city_HKU_en', 'HKU_LATITUDE', 'HKU_LONGITUDE']].copy()
    location_cities = location_cities.set_index('city_HKU_en').sort_index()

    ss_hour_increase_df_new = ss_hour_increase_df.join(location_cities)
    ss_hour_increase_df_new['average_building_height'] = average_building_height
    ss_hour_increase_df_new['facade_roof_ratio'] = facade_roof_ratio
    ss_hour_increase_df_new['facade_area'] = key_information['Area_facade-0(km2)']
    ss_hour_increase_df_level = ss_hour_increase_df_new.copy()
    order = 'ratio'
    if order == 'height':
        bins = [0, 7, 9, float('inf')]
        labels = [0, 1, 2]
        ss_hour_increase_df_level['average_building_height_level'] = pd.cut(ss_hour_increase_df_level['average_building_height'], bins=bins, labels=labels)
        ss_hour_increase_df_level['average_building_height_level'] = ss_hour_increase_df_level['average_building_height_level'].astype(int)
        ss_hour_increase_df_level.sort_values(by=['average_building_height_level', 'HKU_LATITUDE'],ascending=[True, False], inplace=True)
    elif order == 'ratio':
        bins = [0, 1.5, 2.0, float('inf')]
        labels = [0, 1, 2]
        ss_hour_increase_df_level['average_building_height_level'] = pd.cut(ss_hour_increase_df_level['facade_roof_ratio'], bins=bins, labels=labels)
        ss_hour_increase_df_level['average_building_height_level'] = ss_hour_increase_df_level['average_building_height_level'].astype(int)
        ss_sort = ss_hour_increase_df_level.sort_values(by='facade_roof_ratio',ascending=False)
        sorted_index = ss_sort.index.map(lambda x: ss_hour_increase_df_level.index.get_loc(x)).tolist()
        numbers = [(ss_sort['average_building_height_level'] == 2).sum(), (ss_sort['average_building_height_level'] == 1).sum(),(ss_sort['average_building_height_level'] == 0).sum()]
        list_city = [sorted_index[:numbers[0]],sorted_index[numbers[0]:(numbers[0]+numbers[1])],sorted_index[(numbers[0]+numbers[1]):102]]
    return sorted_index,list_city

def regulate(city_name):
    path_type = 'Power'
    path_cap = 'Capacity'
    A_Cap = []
    A_Pow = []
    for Acase in range(6):        
        A1 = path_cap+'/Cap_facade_'+city_name+'.npy'
        AP1 = path_type+str(Acase+1)+'/N_P_facade_ideal_'+str(Acase+1)+'_'+city_name+'.npy'
        AP2 = path_type+str(Acase+1)+'/N_P_facade_'+str(Acase+1)+'_'+city_name+'.npy'

        A_sele = np.where(np.load(A1)[:,-1]-np.load(A1)[:,Acase] > 0)[0]
        A_sele_i = np.where(np.load(A1)[:,-1]-np.load(A1)[:,Acase] <= 0)[0]

        Avg_Cap_id,Avg_Cap = np.sum(np.load(A1)[A_sele,-1]), np.sum(np.load(A1)[A_sele,Acase])
        if Avg_Cap_id <= Avg_Cap:
            print('Bad Cap!')
        Avg_Cap_ratio = Avg_Cap/Avg_Cap_id

        Ac_id,Ac = np.load(A1)[:,-1], np.load(A1)[:,Acase]

        Ac[A_sele_i] = Ac_id[A_sele_i]*Avg_Cap_ratio
        
        A_Cap.append(Ac)

        Aeid,Ae = np.load(AP1), np.load(AP2)
        count1,count2 = 0,0
        for i in range(Ac.shape[0]):
            A_ind1 = np.where(Aeid[i,:] == 0)[0]
            A_ind2 = np.where((Aeid[i,:] > 0) & (Ae[i,:]/Ac[i] > (Aeid[i,:]-Ae[i,:])/(Ac_id[i]-Ac[i])) & ((Aeid[i,:]-Ae[i,:])/(Ac_id[i]-Ac[i]) > 0))[0]
            A_ind3 = np.union1d(A_ind1, A_ind2)
            A_ind4 = np.setdiff1d(np.arange(8760), A_ind3)

            Ae[i,A_ind1] = 0
            Ae[i,A_ind4] = 0.5*(Aeid[i,A_ind4]/Ac_id[i] + Aeid[i,A_ind4]/Ac[i])*Ac[i]
            count1 += len(A_ind2)
            count2 += len(A_ind4)
    
        A_Pow.append(Ae)
        print(count2/(count1+count2))
    return A_Cap,A_Pow

def read_cluster(D_load_cc):
    from sklearn_extra.cluster import KMedoids
    mon_d = [31,28,31,30,31,30,31,31,30,31,30,31]
    EER_c, EER_h = 3.3, 3.0
    Clu_center = np.zeros((102,12),dtype = int)
    city_north = pd.read_excel('City_north_south.xlsx', index_col=0)
    City_statistic = pd.read_excel('City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
    for cc in range(102):
        print(cc)
        city_name = City_statistic.index[cc]
        if city_north.loc[city_name, 'North'] == 0:
            heating_electrification_rate = 1
        else:
            heating_electrification_rate = 0.5
        D_load_sum = D_load_cc[:,0,:]+D_load_cc[:,1,:]*heating_electrification_rate/EER_h+D_load_cc[:,2,:]/EER_c
        D_load_0 = np.sum(D_load_sum,axis = 0)
        for mm in range(12):
            Cluster_load = D_load_0[sum(mon_d[:mm])*24:sum(mon_d[:mm+1])*24].reshape(sum(mon_d[:mm+1])-sum(mon_d[:mm]),24)
            kmedoids = KMedoids(n_clusters=1, random_state=0).fit(Cluster_load)
            center = kmedoids.medoid_indices_[0]
            Clu_center[cc,mm] = int(center)
    return Clu_center


def read_cluster_allyear():
    from sklearn_extra.cluster import KMedoids
    mon_d = [31,28,31,30,31,30,31,31,30,31,30,31]
    mon_d_new = [31,31,28,31,30,31,30,31,31,30,31,30]
    
    EER_c, EER_h = 3.3, 3.0
    Clu_center = np.zeros((102,12),dtype = int)
    Clu_days = np.zeros((102,12))    
    city_north = pd.read_excel('City_north_south.xlsx', index_col=0)
    City_statistic = pd.read_excel('City_statistic.xlsx',sheet_name = 'Class_Volume', index_col=0)
    for cc in range(102):
        print(cc)
        city_name = City_statistic.index[cc]
        
        if city_north.loc[city_name, 'North'] == 0:
            heating_electrification_rate = 1
        else:
            heating_electrification_rate = 0.5
        D_load_cc = np.load(city_name+'_hybrid.npy')/1e0
        D_load_sum = D_load_cc[:,0,:]+D_load_cc[:,1,:]*heating_electrification_rate/EER_h+D_load_cc[:,2,:]/EER_c
        D_load_0 = np.sum(D_load_sum,axis = 0).reshape(365,24)
        
        st = [0,sum(mon_d_new[:3]),sum(mon_d_new[:6]),sum(mon_d_new[:9])]
        ed = [sum(mon_d_new[:3]),sum(mon_d_new[:6]),sum(mon_d_new[:9]),sum(mon_d_new[:12])]
        for mm in range(4):
            D_load_new = np.concatenate([D_load_0[sum(mon_d[:11]):,:],D_load_0[:sum(mon_d[:11]),:]], axis = 0)
            Cluster_load = D_load_new[st[mm]:ed[mm],:]
            kmedoids = KMedoids(n_clusters=3, random_state=0).fit(Cluster_load)
            cluster_labels = kmedoids.labels_
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                Clu_days[cc,3*mm+label] = count
            for cen in range(len(kmedoids.medoid_indices_)):
                center = kmedoids.medoid_indices_[cen]
                if mm == 0 and center < 31:
                    Clu_center[cc,3*mm+cen] = int(center+sum(mon_d[:11]))
                elif mm == 0 and center >= 31:
                    Clu_center[cc,3*mm+cen] = int(center-31)
                elif mm == 1:
                    Clu_center[cc,3*mm+cen] = int(center + sum(mon_d[:2]))
                elif mm == 2:
                    Clu_center[cc,3*mm+cen] = int(center + sum(mon_d[:5]))
                elif mm == 3:
                    Clu_center[cc,3*mm+cen] = int(center + sum(mon_d[:8]))
    return Clu_center,Clu_days
