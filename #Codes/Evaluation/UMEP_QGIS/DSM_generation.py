
import numpy as np
import pandas as pd

City_name = ['Beijing', 'Changchun', 'Changsha', 'Chengdu', 'Chongqing', 'Fuzhou', 'Guangzhou', 'Guiyang', 'Haerbin', 'Haikou', 'Hangzhou', 'Hefei', 'Huhehaote', 'Jinan', 'Kunming', 'Lanzhou', 'Nanchang', 'Nanjing', 'Nanning', 'Shanghai', 'Shenyang', 'Shijiazhuang', 'Taiyuan', 'Tianjin', 'Wuhan', 'Wulumuqi', 'Xian', 'Xining', 'Yinchuan', 'Zhengzhou']

# path = 'D:/OneDrive/HKU phD/Direction-Urban G/##HH-Building win-PV/Data-simulation/30_simu_cities/'
# path_file = 'D:\\OneDrive\\HKU phD\\Direction-Urban G\\##HH-Building win-PV\\Data-simulation\\30_simu_cities\\'

for cc in City_name:
    print(cc)

    df_read = np.array(pd.read_excel(cc+'\\'+cc+'_sample_DEM_coordinate'+'.xlsx'))

    for i in range(1,65):
        print(i)
        [xmin,xmax,ymin,ymax] = df_read[i-1,:]
        xmax = xmin+2199
        ymin = ymax-2199
        processing.run("umep:Spatial Data: DSM Generator", {'INPUT_DEM':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_DEM.TIF','INPUT_POLYGONLAYER':cc+'/'+cc+'_'+str(i)+'/'+str(i)+'.shp','INPUT_FIELD':'height','USE_OSM':False,'BUILDING_LEVEL':3.1,'EXTENT':str(xmin)+','+str(xmax)+','+str(ymin)+','+str(ymax)+' [EPSG:3857]','PIXEL_RESOLUTION':3,'OUTPUT_DSM':cc+'/'+cc+'_'+str(i)+'/'+cc+'_'+str(i)+'_DSM.tif'})
        print('finish')

#    processing.run("umep:Spatial Data: DSM Generator", {'INPUT_DEM':'D:/OneDrive/HKU phD/Direction-Urban G/##HH-Building win-PV/Data-simulation/30_simu_cities/Beijing/Beijing_'+str(i)+'/Beijing_'+str(i)+'_DEM.TIF','INPUT_POLYGONLAYER':'D:/OneDrive/HKU phD/Direction-Urban G/##HH-Building win-PV/Data-simulation/30_simu_cities/Beijing/Beijing_'+str(i)+'/Beijing_'+str(i)+'.shp','INPUT_FIELD':'height','USE_OSM':False,'BUILDING_LEVEL':3.1,'EXTENT':str(xmin)+'12946831.453700000,12949060.388800001,4848660.212000000,4851550.862300000 [EPSG:3857]','PIXEL_RESOLUTION':3,'OUTPUT_DSM':'D:/OneDrive/HKU phD/Direction-Urban G/##HH-Building win-PV/Data-simulation/30_simu_cities/Beijing/Beijing_'+str(i)+'/Beijing_'+str(i)+'_DSM.tif'})