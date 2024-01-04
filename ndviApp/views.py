from PIL import Image

from osgeo import gdal
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
np.seterr(divide='ignore', invalid='ignore')

class Grid:
    # =============================================================================
    #     1. 函数1 read_tiff读取影像
    # =============================================================================
    def read_tiff(self, filename):
        # 读取影像，获取影像位置
        dataset = gdal.Open(filename)
        # 获取影像波段数,获取影像长宽
        im_band=dataset.GetRasterBand   #波段数
        im_width = dataset.RasterXSize  # 宽，列数
        im_height = dataset.RasterYSize  # 高,行数
        # 仿射矩阵
        im_GeoTransform = dataset.GetGeoTransform()
        # 地图投影信息
        img_proj = dataset.GetProjection()

        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
        del dataset
        return img_proj, im_GeoTransform, im_data

    # =============================================================================
    #     函数2 write_tiff，存储为GTIFF
    # =============================================================================
    def write_tiff(self, filename, im_proj, im_GeoTransform, im_data):
        # 判断栅格数据类型
        if 'int8' in im_data.dtype.name:  # int8 uint8
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:  # int16 uint16~DN值/反射率
            datatype = gdal.GDT_UInt16
        elif 'int32' in im_data.dtype.name:  # int16 uint16~DN值/反射率
            datatype = gdal.GDT_UInt32
        else:
            datatype = gdal.GDT_Float32
            # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape
            # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_GeoTransform)
        dataset.SetProjection(im_proj)
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset

    # =============================================================================
    #     函数3 calndvi，输入波段数组，返回单波段数组
    # =============================================================================
    def calndvi(self, im_data):
        band_r = im_data[1, :, :]#band4 R
        band_nir = im_data[2, :, :]#band5 NIR
        ndvi = (band_nir - band_r) / (band_nir + band_r)
        nan_index = np.isnan(ndvi)
        ndvi[nan_index] = 0
        sc = plt.imshow(ndvi, cmap=plt.cm.jet)  # 设置cmap为RGB图
        plt.colorbar()  # 显示色度条
        plt.savefig('media/ndvi/result.png',dpi=300)
        plt.show()

        return ndvi
@login_required
def ndvi(request):
    filepath = r'media/ndvi/'
    filename = r'3.tif'
    # tiff_image = Image.open(filepath+filename)
    # png_image = tiff_image.convert('RGBA')
    # png_image.save(filepath+'input_image.png', 'PNG')
    run=Grid()
    img_proj, im_GeoTransform, im_data=run.read_tiff(filepath+filename)
    ndvi=run.calndvi(im_data)
    print(ndvi)
    run.write_tiff('media/ndvi/nvdi_result.tiff', img_proj, im_GeoTransform, ndvi)
    return render(request, 'welcome.html',
                  { 'in_image_url': filepath+'input_image.png', 'out_image_url': filepath+'result.png'})
