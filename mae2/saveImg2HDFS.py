from hdfs import InsecureClient


# 连接到HDFS，将IP地址更换为自己HDFS服务的IP地址，user更换为自己HDFS的用户名
client = InsecureClient('http://192.168.56.109:50070', user='hadoop')


# 创建目录
def mkdirs(client, hdfs_path):
    client.makedirs(hdfs_path)
    hdfs_file_status = client.status(hdfs_path)
    print(f"File status: {hdfs_file_status}")


# 删除hdfs文件
def delete_hdfs_file(client, hdfs_path):
    client.delete(hdfs_path)
    hdfs_file_status = client.status(hdfs_path)
    print(f"File status: {hdfs_file_status}")


# 上传文件到hdfs
def put_to_hdfs(client, local_path, hdfs_path):
    client.upload(hdfs_path, local_path, cleanup=True)
    hdfs_file_status = client.status(hdfs_path)
    print(f"File status: {hdfs_file_status}")


# 从hdfs获取文件到本地
def get_from_hdfs(client, hdfs_path, local_path):
    client.download(hdfs_path, local_path, overwrite=False)
    hdfs_file_status = client.status(hdfs_path)
    print(f"File status: {hdfs_file_status}")


# 追加数据到hdfs文件
def append_to_hdfs(client, hdfs_path, data):
    client.write(hdfs_path, data, overwrite=False, append=True)
    hdfs_file_status = client.status(hdfs_path)
    print(f"File status: {hdfs_file_status}")


# 覆盖数据写到hdfs文件
def write_to_hdfs(client, hdfs_path, data):
    client.write(hdfs_path, data, overwrite=True, append=False, n_threads=5)
    hdfs_file_status = client.status(hdfs_path)
    print(f"File status: {hdfs_file_status}")


# 返回目录下的文件
def list(client, hdfs_path):
    return client.list(hdfs_path, status=False)

#例如可以将NDVI的结果result.png存储到HDFS上根目录下"/",可以调用以下函数：
#  put_to_hdfs(client,'./result.png','/')


# HDFS目标文件夹路径 更换为自己的HDFS文件夹的路径
hdfs_folder_path = '/path/on/hdfs/images/'

image_path_list = []
# 将image_path_list中图片数据存储到HDFS上，循环处理每张图像并上传到 HDFS
for index, image_path in enumerate(image_path_list):

    # HDFS目标路径
    hdfs_image_path = f'{hdfs_folder_path}image_{index}.png'

    # 将本地图像上传到 HDFS
    write_to_hdfs(client,hdfs_image_path,image_path)

####返回目录下的所有文件
def list_all_files_and_folders(client, hdfs_path):
    all_items = []

    def list_recursive(client, hdfs_path):
        hdfs_items = client.list(hdfs_path, status=False)
        for item in hdfs_items:
            full_path = f'{hdfs_path}/{item}'
            all_items.append(full_path)
            if client.status(full_path)['type'] == 'DIRECTORY':
                list_recursive(client, full_path)

    list_recursive(client, hdfs_path)
    return all_items

if __name__ == '__main__':
    client = InsecureClient('http://192.168.108.128:50070', user='hadoop')
    # 上传文件
    # put_to_hdfs(client,'./train','/')
    #mkdirs(client,'/train_log/')
    # print(list(client,'/'))
    #delete_hdfs_file(client,'/log')
    all_items = list_all_files_and_folders(client, '/')
    for item in all_items:
        print(item)

    # 拉区文件
    # get_from_hdfs(client,'/1.tif','./')
    # mkdirs(client,'/student/')