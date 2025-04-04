#%%
import pandas as pd
import numpy as np
import Functions as Functions

cities_hku_dest = pd.read_hdf('cities_hku_dest.hdf', key='cities_hku_dest')

loads_diff_city_diff_building_type_per_area = pd.read_hdf('loads_diff_city_diff_building_type_per_area.hdf',
        key='loads_diff_city_diff_building_type_per_area')

file_path = 'City_statistic.xlsx'

building_volume_df = pd.read_excel(file_path, index_col=0)  

building_volume_df = building_volume_df.drop(columns='Total(km3)')

building_volume_df.columns = building_volume_df.columns.astype(str)

missing_cities = list(building_volume_df[building_volume_df.sum(axis=1) == 0].index)
missing_cities += ['Baoding', 'Daqing', 'Changchun']

land_use_mapping = {
    "101": "住宅用地",
    "201": "商业办公",
    "202": "商贸服务",
    "301": "工业用地",
    "401": "道路",
    "402": "运输场站",
    "403": "机场用地",
    "501": "机关团体用地",
    "502": "教育科研用地",
    "503": "医疗用地",
    "504": "体育文化",
    "505": "公园与绿地"
}

land_use_to_building_types = {
    "住宅用地": ["Terraced house", "Low-rise", "High-rise (slab-type)", "High-rise (tower-type)"],
    "商业办公": ["Commercial office A", "Commercial office B"],
    "商贸服务": ["Small hotel", "Large hotel", "Shopping mall"],
    "机关团体用地": ["Government office A", "Government office B"],
    "教育科研用地": ["Primary/secondary school", "University"],
    "医疗用地": ["Outpatient", "Inpatient"],

    "道路": ["Small hotel", "Large hotel", "Shopping mall"],
    "运输场站":  ["Small hotel", "Large hotel", "Shopping mall"],
    "机场用地": ["Small hotel", "Large hotel", "Shopping mall"],
    "体育文化": ["Small hotel", "Large hotel", "Shopping mall"],
    "公园与绿地": ["Terraced house", "Low-rise", "High-rise (slab-type)", "High-rise (tower-type)"],
}

building_types_to_abbreviations = {
    "Terraced house": "Th",    
    "Low-rise": "Low",        
    "High-rise (slab-type)": "HighS", 
    "High-rise (tower-type)": "HighT",
    "Government office A": "GoA",  
    "Government office B": "GoB",  
    "Commercial office A": "CoA",  
    "Commercial office B": "CoB", 
    "Primary/secondary school": "Sch", 
    "University": "Uni",               
    "Outpatient": "Outp",              
    "Inpatient": "Inp",              
    "Small hotel": "SH",         
    "Large hotel": "LH",     
    "Shopping mall": "Mall"  
}


abbreviation_to_coefficients = {
    "Th": 0.1,  
    "Low": 0.3,
    "HighS": 0.3,
    "HighT": 0.3,

    "CoA": 0.8,  
    "CoB": 0.2,  

    "SH": 0.1,  
    "LH": 0.2,  
    "Mall": 0.7,

    "GoA": 0.8, 
    "GoB": 0.2, 

    "Sch": 0, 
    "Uni": 1, 


    "Outp": 0.2, 
    "Inp": 0.8,  
}

abbreviation_to_story_heights = {
    "Th": 3,  
    "Low": 3, 
    "HighS": 3,
    "HighT": 3, 

    "CoA": 3,
    "CoB": 3,

    "SH": 3.0,
    "LH": 3.0,
    "Mall": 3,

    "GoA": 3,
    "GoB": 3,

    "Sch": 3.0,
    "Uni": 3.0,

    "Outp": 3.0,
    "Inp": 3,
}

city_north = pd.read_excel('City_north_south.xlsx', index_col=0)



for i_city in range(102):
    city_hku = building_volume_df.index[i_city]
    if i_city in [97,92,2,6,8,69]:
        i_vol = 8
    else:
        i_vol = 6

    city_dest = cities_hku_dest[cities_hku_dest['city_HKU_en'] == city_hku].iloc[0]['city_DeST']

    print(f'============================== #{i_city}, {city_hku}, {city_dest}==============================')

    G_type = np.load('Grid_type_'+city_hku+'.npy')
    Feas_read_sta = np.load(city_hku+'_ALL_Featuers.npy')[:,[i for i in range(14)]+[15,16]]
    indices_non_zero = np.where(Feas_read_sta[:,11] != 0)[0]

    loads = [np.zeros((G_type.shape[0],3,8760)),np.zeros((G_type.shape[0],3,8760)),np.zeros((G_type.shape[0],3,8760))]

    if i_city in [0,4,10,11,1,3,93,94,5,13,12,7,9]:
        Feas_read_info = np.load(city_hku+'_ALL_Featuers_supplementary.npy')
    else:
        Feas_read_info = np.load(city_hku+'_ALL_Featuers.npy')[:,range(17,29)]
    for i in range(G_type.shape[0]):
        if i % 1000 == 0:
            print(i)
        column_list = ['101', '201', '202', '301', '401', '402', '403', '501', '502', '503', '504', '505']
        building_volume_df_new = pd.DataFrame(np.zeros((1,12)), columns=column_list)
        building_volume_df_new.iloc[0,column_list.index(str(int(G_type[i,0])))] = Feas_read_info[indices_non_zero,:][i,i_vol]

        land_use_code_real = str(int(G_type[i,0]))    
        building_volume_land_use = building_volume_df_new.loc[0, land_use_code_real]  # * 1000**3
        
        for land_use_code_ii in range(len(['101','201','202'])):
            land_use_code = ['101','201','202'][land_use_code_ii]
            land_use_type = land_use_mapping[land_use_code]
            if land_use_type not in land_use_to_building_types:
                continue

            corresponding_building_types = land_use_to_building_types[land_use_type]

            loads_city_list_ele,loads_city_list_heat,loads_city_list_cool = [],[],[]
            for building_type in corresponding_building_types:

                building_type_abbreviation = building_types_to_abbreviations[building_type]

                building_volume_coefficient = abbreviation_to_coefficients[building_type_abbreviation]

                story_height = abbreviation_to_story_heights[building_type_abbreviation]

                building_volume = building_volume_land_use * building_volume_coefficient
                building_area = building_volume / story_height

                loads_per_area_df = loads_diff_city_diff_building_type_per_area\
                    .loc[city_hku][land_use_code+building_type_abbreviation]

                loads_per_area_df.index = pd.MultiIndex.from_product(
                        [[city_hku], loads_per_area_df.index],
                        names=['location', 'timestamp'])

                if land_use_code == '101' and loads_per_area_df['electricity'].sum()<0.8*loads_per_area_df['heat'].sum():
                    real_ele = loads_per_area_df['electricity']
                else:
                    real_ele = loads_per_area_df['electricity']

                total_electricity_load = (real_ele) * building_area
                total_heat_load = (loads_per_area_df['heat']) * building_area
                total_cool_load = (loads_per_area_df['cool']) * building_area

                total_electricity_load.rename(land_use_code_real+building_type_abbreviation, inplace=True)
                total_heat_load.rename(land_use_code_real+building_type_abbreviation, inplace=True)
                total_cool_load.rename(land_use_code_real+building_type_abbreviation, inplace=True)
                
                loads_city_list_ele.append(total_electricity_load)
                loads_city_list_heat.append(total_heat_load)
                loads_city_list_cool.append(total_cool_load)
        
            loads[land_use_code_ii][i,0,:] = np.sum(pd.concat(loads_city_list_ele, axis=1).values,axis=1)
            loads[land_use_code_ii][i,1,:] = np.sum(pd.concat(loads_city_list_heat, axis=1).values,axis=1)
            loads[land_use_code_ii][i,2,:] = np.sum(pd.concat(loads_city_list_cool, axis=1).values,axis=1)
    
    EER_c, EER_h = 3.3, 3.0
    A_load_cc = 0.8*loads[0]+0.1*loads[1]+0.1*loads[2]
    A_load_sum = A_load_cc[:,0,:] + A_load_cc[:,2,:]/EER_c + 0.45*(1/EER_h)*A_load_cc[:,1,:]
    print(np.sum(A_load_sum[:,:])/1e9)
    np.save(city_hku+'_hybrid.npy',0.8*loads[0]+0.1*loads[1]+0.1*loads[2])
