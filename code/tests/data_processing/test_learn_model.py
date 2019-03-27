from sklearn.cluster import KMeans

from myo_sign_language.data_processing.learn_model import get_overlapped_chunks_separated_for_files
from myo_sign_language.data_processing.learn_model import get_all_overlapped_chunks
from myo_sign_language.data_processing.learn_model import get_histogram_basic_on_kmean


def test_get_overlapped_chunks_separated_for_files():
    sensor1 = [212, 1212, 4545]
    sensor2 = [612, 6212, 6545]
    file = [sensor1, sensor2]
    files_as_nested_list = [file, file]
    windows_size = 2
    overlaped = 1
    windows = get_overlapped_chunks_separated_for_files(files_as_nested_list, windows_size, overlaped)
    windowed_sensor1 = [[212, 1212], [1212, 4545]]
    windowed_sensor2 = [[612, 6212], [6212, 6545]]
    chunked_file = [windowed_sensor1, windowed_sensor2]
    assert windows == [chunked_file, chunked_file]

def test_get_overlapped_chunks_separated_for_files_when_windows_size_too_big():
    sensor1 = [212, 1212, 4545]
    sensor2 = [612, 6212, 6545]
    file = [sensor1, sensor2]
    files_as_nested_list = [file, file]
    windows_size = 5
    overlaped = 1
    windows = get_overlapped_chunks_separated_for_files(files_as_nested_list, windows_size, overlaped)
    windowed_sensor1 = [[212, 1212], [1212, 4545]]
    windowed_sensor2 = [[612, 6212], [6212, 6545]]
    chunked_file = [windowed_sensor1, windowed_sensor2]
    assert windows == [chunked_file, chunked_file]


def test_get_all_overlapped_chunks():
    sensor1 = [212, 1212, 4545]
    sensor2 = [812, 1912, 1545]
    file = [sensor1, sensor2]
    files_as_nested_list = [file, file]
    windows_size = 2
    overlaped = 1
    windows = get_all_overlapped_chunks(files_as_nested_list, windows_size, overlaped)
    windowed_sensor1_file_1 = [[212, 1212], [1212, 4545]]
    windowed_sensor2_file_1 = [[812, 1912], [1912, 1545]]
    windowed_sensor1_file_2 = [[212, 1212], [1212, 4545]]
    windowed_sensor2_file_2 = [[812, 1912], [1912, 1545]]
    assert len(windows) == 2
    assert windows == [
        windowed_sensor1_file_1+windowed_sensor1_file_2,
        windowed_sensor2_file_1+windowed_sensor2_file_2]

def test_get_all_overlapped_chunks_when_windows_size_too_big():
    sensor1 = [212, 1212, 4545]
    sensor2 = [812, 1912, 1545]
    file = [sensor1, sensor2]
    files_as_nested_list = [file, file]
    windows_size = 4
    overlaped = 1
    windows = get_all_overlapped_chunks(files_as_nested_list, windows_size, overlaped)
    windowed_sensor1_file_1 = [[212, 1212], [1212, 4545]]
    windowed_sensor2_file_1 = [[812, 1912], [1912, 1545]]
    windowed_sensor1_file_2 = [[212, 1212], [1212, 4545]]
    windowed_sensor2_file_2 = [[812, 1912], [1912, 1545]]
    assert len(windows) == 2
    assert windows == [
        windowed_sensor1_file_1+windowed_sensor1_file_2,
        windowed_sensor2_file_1+windowed_sensor2_file_2]

# def test_get_histogram_basic_on_kmean():
#     clusters_number = 5
#     file = [
#         [(1, 2, 3), (3, 4, 5), (5, 6, 7)], [(9, 10, 11), (11, 12, 13), (13, 14, 15)]
#     ]
#
#     files_as_nested_list = [file, file]
#     learnign_clusters = [(1, 1, 1,), (2, 2, 2,), (3, 3, 3,), (4, 4, 4,), (5, 5, 5,), (6, 6, 6,)]
#
#     kmeans_models = [
#         KMeans(n_clusters=clusters_number).fit(learnign_clusters),
#         KMeans(n_clusters=clusters_number).fit(learnign_clusters)
#     ]
#     histogram = get_histogram_basic_on_kmean(clusters_number, kmeans_models, files_as_nested_list)
#     # every_sensor * number_of clusters
#     assert len(histogram[0]) == len(file) * clusters_number
#     assert len(histogram[1]) == len(file) * clusters_number
