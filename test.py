import os
import numpy as np
import pandas as pd
from scipy import stats
from process_util import Config
from process_util import Process


def test():
    """Perform test for synthetic data
    """
    np.random.seed(42)
    shape = (7,3)
    feat = np.random.randint(-100, 100, shape)
    expected_z_score = stats.zscore(feat, axis=0)
    argmax = feat.argmax(axis=1)
    expected_z_score = np.hstack((argmax.reshape(argmax.shape[0],1), expected_z_score))
    max_feat = feat.max(axis=1)
    mean_values = np.take(feat.mean(axis=0), argmax)
    abs_diff = np.abs(max_feat - mean_values)
    expected_z_score = np.insert(expected_z_score, [1], abs_diff.reshape((abs_diff.shape[0],1)),axis=1)

    feat_list = feat.tolist()
    feat_list_str = list(map(lambda x: ','.join(map(str, x)), feat_list))
    feat_list_str = list(map(lambda x: ','.join(['2', x]),feat_list_str))
    test_df = pd.DataFrame({'id_job':np.arange(shape[0]), 'features':feat_list_str})

    class CustomConfig(Config):
        INPUT_FILE_PATH = 'simple_test.tsv'
        OUTPUT_FILE_PATH = 'simple_proc.tsv'
        CHUNK_SIZE = 30
        
    my_config = CustomConfig()

    try:
        os.remove(my_config.INPUT_FILE_PATH)
    except OSError:
        pass 
    try:
        os.remove(my_config.OUTPUT_FILE_PATH)
    except OSError:
        pass 

    test_df.to_csv(my_config.INPUT_FILE_PATH, index=False, sep='\t')  
    process = Process(my_config)
    process.process_data()

    proc_test = pd.read_csv(my_config.OUTPUT_FILE_PATH, sep='\t')
    actual_z_score = proc_test.loc[:, proc_test.columns != 'id_job']
    print('-'*10 + 'expected' + '-'*10)
    print(expected_z_score)
    print('-'*10 + 'actual' + '-'*10)
    print(actual_z_score.values)

    assert np.allclose(expected_z_score, actual_z_score.values) == True
