# 1. Extract the content of the `20news-bydate.tar` file:

import tarfile

tar_file_path = '20news-bydate.tar'  # Replace with the path to your .tar file
tar = tarfile.open(tar_file_path)
tar.extractall()
tar.close()
