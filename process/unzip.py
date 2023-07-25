import os
import shutil
import zipfile
from os.path import join, getsize


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:   
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)   
        
            # print(dst_dir)
            # print(file)
            shutil.move(os.path.join(dst_dir, file), dst_dir)
            # shutil.rmtree(os.path.join(dst_dir, file))

    else:
        print('This is not zip')



if __name__ == '__main__':
    file = '/mnt/traffic/data/deepfake/DeeperForensics-1.0/sliced/manipulated_videos/reenact_postprocess'
    out_file = '/mnt/traffic/home/shuaichao/data/FF++/c23/data/DeeperForensics'

    if not os.path.exists(out_file):
        os.mkdir(out_file)
    videos = os.listdir(file)

    for vid in videos:
        zip_src = os.path.join(file, vid, 'frames.zip')
        dst_dir = os.path.join(out_file, vid)
        if os.path.exists(dst_dir):
            continue
        print(vid)
        unzip_file(zip_src, dst_dir)
        