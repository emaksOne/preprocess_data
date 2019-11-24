"""
Usage: import the module, or run from
       the command line as such:
       
    python3 process_util.py --input=/path/to/input/file --output=/path/to/output/file --chunksize=chunksize
"""


import os
import sys
import itertools
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

DEFAULT_TEMP_SQLITE_DB_NAME = 'temp.db'
DEFAULT_INPUT_FILE_PATH = 'data/test.tsv'
DEFAULT_OUTPUT_FILE_PATH = 'data/test_proc.tsv'
DEFAULT_CHUNK_SIZE = 500
DEFAULT_SQLITE_DB_TEMP_NAME = 'temp.db'


class Config:
    """Config class for settin input path, output path, chunk size, name of temp db
    Derive from this class to override some values
    """
    INPUT_FILE_PATH = DEFAULT_INPUT_FILE_PATH
    OUTPUT_FILE_PATH = DEFAULT_OUTPUT_FILE_PATH
    CHUNK_SIZE = DEFAULT_CHUNK_SIZE
    SQLITE_DB_TEMP_NAME = DEFAULT_SQLITE_DB_TEMP_NAME


class Process:
    """Class for preprocessing. 
    Usage: 
        config = Config()
        process = Process(config)
        process.process_data()
    """
    def __init__(self, config):
        self._config = config
        self._n = 0

        ## TODO: refactor this if we have new attribute type with different size
        first_row = pd.read_csv(config.INPUT_FILE_PATH, nrows=1, sep='\t')
        features_str = first_row['features'].values[0]
        feature_size = len(features_str.split(','))-1
        
        # track means of attrs for update by chunks
        self._means = np.zeros(feature_size)
        # track means of square attrs for update by chunks
        self._sq_means =np.zeros(feature_size)
        # track std of attrs for update by chunks
        self._std = np.zeros(feature_size)
        
        ## TODO: extract feature types dynamicaly
        self._groups = [features_str.split(',')[0]]
       
        self._feature_names = []
        self._featur_stat_names = []
        

    def process_data(self):
        """This public method. Process file specified in config by chunks
        """
        try:
            #remove temp.db if exist
            try:
                os.remove(self._config.SQLITE_DB_TEMP_NAME)
            except OSError:
                pass 
            self._disk_engine = create_engine(f'sqlite:///{self._config.SQLITE_DB_TEMP_NAME}')
            self._conn = self._disk_engine.connect()
            rows_count = 0
            for chunk_df in pd.read_csv(self._config.INPUT_FILE_PATH, chunksize=self._config.CHUNK_SIZE, sep='\t'):
                self.process_chunk(chunk_df)
                rows_count += len(chunk_df)
                print(f'Processed {rows_count} rows')

            self.db_to_csv()
        finally:
            self.dispose()

    def process_chunk(self, chunk_df):
        """Method perform process logic on chunk dataframe
        # Arguments:
            chunk_df: pandas dataframe
        """
        chunk_transformed = self.transform_columns(chunk_df)
        
        self._feature_names = chunk_transformed.loc[:, chunk_transformed.columns != 'id_job'].columns.values
        
        chunk_transformed = self.append_max_feature_atr_index(chunk_transformed)
        
        new_statistics = self.find_new_statistics(chunk_transformed)
        new_means, new_sq_means, new_std = new_statistics

        chunk_transformed = self.append_abs_diff(chunk_transformed, new_statistics)
        chunk_transformed = self.populate_with_z_score(chunk_transformed, new_statistics)

        ## TODO: extend with new type of scores

        self.update_previous_data_with_new_statistics(new_statistics)
        self.append_new_data(chunk_transformed)

        self._means = new_means
        self._sq_means = new_sq_means
        self._std = new_std
        self._n += len(chunk_df)

    def append_max_feature_atr_index(self, df):
        """Find max feature index per row
        # Arguments:
            df: pandas dataframe
        # Returns:
            df: dataframe with new max_feature_{group}_index column 
        """
        for group in self._groups:
            filter_col = [col for col in df if col.startswith(f'features_{group}')]
            max_feature_col_name = df[filter_col].idxmax(axis=1)
            max_feature_ind = list(map(lambda x: x[2], max_feature_col_name.str.split('_')))
            df[f'max_feature_{group}_index'] = max_feature_ind

        return df

    def append_abs_diff(self, df, new_statistics):
        """Find diff between max element in atrribut with corresponding attr mean value
        # Arguments:
            df: pandas dataframe
            new_statistics: tuple that contains means for attrs from start to current chunk included, 
                            means of squared elemetns for attrs from start to current chunk included, 
                            std for attrs from start to current chunk included
        # Returns:
            df: dataframe with max_feature_{group}_abs_mean_diff column 
        """
        new_means = new_statistics[0]
        for group in self._groups:
            filter_col = [col for col in df if col.startswith(f'features_{group}')]
            max_feature_ind = df[f'max_feature_{group}_index']
            max_feature = df[filter_col].max(axis=1)
            mean_values = np.take(new_means, max_feature_ind)
            #abs operation permorm in the end. Because restore va
            df[f'max_feature_{group}_abs_mean_diff'] = max_feature - mean_values

        return df
        
    def transform_columns(self, df):
        """Transform features column (2,324,423,423,234, ...) 
        to multiple columns 'features_2_{i}'
        # Arguments:
            df: pandas dataframe
        # Returns: 
            df: dataframe with new columns
        """
        df_transformed = pd.DataFrame()
        df_transformed['id_job'] = df['id_job']
        
        ## TODO: refactor in case of new attr types  
        ## (I don't know how we add new type. In new rows or new col or in the current cell. So made it as simple as possible)
        temp = df['features'].str.split(",", expand = True) 
        for col_ind in range(temp.shape[1]):
            col_name = f'features_{temp.iloc[0, 0]}_{col_ind - 1}'
            if col_ind > 0:
                df_transformed[col_name] = temp.iloc[: , col_ind]
                df_transformed[col_name] = df_transformed[col_name].astype(str).astype('int64')

        return df_transformed

    def find_new_statistics(self, df):
        """This methods finds mean of attrs, mean of square attrs, std of attr 
        based on previous values _means, _sq_means, _std
        # Arguments:
            df: pandas dataframe
        # Returns
            new_statistics: tuple that contains means for attrs from start to current chunk included, 
                            means of squared elemetns for attrs from start to current chunk included, 
                            std for attrs from start to current chunk included
        """
        n_cur = len(df)
        features = df[self._feature_names].values
        new_means = self._means + (features.mean(axis=0) - self._means)/((self._n + n_cur)/n_cur)
        new_sq_means = self._sq_means + (np.square(features).mean(axis=0) - self._sq_means)/((self._n + n_cur)/n_cur)
        new_std = np.sqrt(new_sq_means - np.square(new_means))
        new_statistics = (new_means, new_sq_means, new_std)

        return new_statistics
        
    
    def populate_with_z_score(self, df, new_statistics):
        """Find z score for given feature attrs and replace it
        # Arguments: 
            df: pandas dataframe
            new_statistics: tuple that contains means for attrs from start to current chunk included, 
                            means of squared elemetns for attrs from start to current chunk included, 
                            std for attrs from start to current chunk included
        # Returns: 
            df: dataframe with 'features_{group}_stand_{attr}' columns
        """
        features = df[self._feature_names].values
        new_means, new_sq_means, new_std = new_statistics
        z_score_for_new = (features - new_means)/new_std

        def transform_name(name):
            parts = name.split('_')
            return f'{parts[0]}_{parts[1]}_stand_{parts[2]}'

        new_names = map(transform_name, self._feature_names)
        self._featur_stat_names = list(new_names)
        z_score_df = pd.DataFrame(z_score_for_new, columns=self._featur_stat_names, index=df.index)
        df = pd.concat([df, z_score_df], axis=1)
        df = df.drop(self._feature_names, axis=1)
      
        return df
      
    def update_previous_data_with_new_statistics(self, new_statistics):
        """Update previously processed chunks that stored in sqlite db with new zcore and 
        diff between max value in the attribute and corresponding mean
        # Arguments:  new_statistics: tuple that contains means for attrs from start to current chunk included, 
                                        means of squared elemetns for attrs from start to current chunk included, 
                                        std for attrs from start to current chunk included            
        """
        if os.path.isfile(self._config.SQLITE_DB_TEMP_NAME) and self._disk_engine.has_table('temp'):
            new_means, new_sq_means, new_std = new_statistics
            
            for chunk_df in pd.read_sql_query('SELECT * FROM temp', con=self._conn, chunksize=self._config.CHUNK_SIZE):
                # restore value using std and mean on previous step
                chunk_df[self._featur_stat_names] = chunk_df[self._featur_stat_names] * self._std + self._means
                # find new z score with new mean and std
                chunk_df[self._featur_stat_names] = (chunk_df[self._featur_stat_names] - new_means)/new_std

                #update abs diff between max value in the attribute and corresponding mean
                for group in self._groups:
                    max_feature_ind = chunk_df[f'max_feature_{group}_index']
                    mean_values = np.take(self._means, max_feature_ind)
                    new_mean_values = np.take(new_means, max_feature_ind)
                    
                    #restore previous value
                    chunk_df[f'max_feature_{group}_abs_mean_diff'] = chunk_df[f'max_feature_{group}_abs_mean_diff'] + pd.Series(mean_values)
                    
                    #update with new mean. Do not calculate abs because it is imposible to restore then. 
                    #Abs operation will be in the end
                    chunk_df[f'max_feature_{group}_abs_mean_diff'] = chunk_df[f'max_feature_{group}_abs_mean_diff'] - pd.Series(new_mean_values)
                
                # here is a workaround. I place this updated chunk on the new buffer table because I 
                # didn't find a way to perform multiple updates sql statements at once.
                # So I simply put updted chunk to another table and remove old chunk from origin table  
                chunk_df.to_sql('temp_2', con=self._conn, if_exists='append', index=False)
                ids = ','.join(map(str, chunk_df['id_job'].values))
                sql = f'DELETE from temp WHERE id_job in ({ids});'
                self._conn.execute(sql)

            # after we process all chunks there will be empty origin table. So remove it.
            # And rename our buffer table as origin
            self._conn.execute('DROP TABLE temp')
            self._conn.execute('ALTER TABLE temp_2 rename to temp')

    def append_new_data(self, df):
        """Append new chunk to sqlite db
        # Arguments:
            df: pandas dataframe
        """
        df.to_sql('temp', self._disk_engine, if_exists='append', index=False)

    def db_to_csv(self):
        """Convert sqlite db to tsv file and perform abs operation for max_feature_{group}_abs_mean_diff column
        """
        for chunk_df in pd.read_sql_query('SELECT * FROM temp', con=self._conn, chunksize=self._config.CHUNK_SIZE):
            # perfomr abs operation
            for group in self._groups:
                chunk_df[f'max_feature_{group}_abs_mean_diff'] = np.abs(chunk_df[f'max_feature_{group}_abs_mean_diff'])
            if not os.path.isfile(self._config.OUTPUT_FILE_PATH):
                chunk_df.to_csv(self._config.OUTPUT_FILE_PATH, header='column_names', index=False, sep='\t')
            else: # else it exists so append without writing the header
                chunk_df.to_csv(self._config.OUTPUT_FILE_PATH, mode='a', header=False, index=False, sep='\t')

    def delete_temp_db(self):
        """Remove temprorary sqlite db
        """
        try:
            os.remove(self._config.SQLITE_DB_TEMP_NAME)
        except OSError:
            pass 

    def dispose(self):
        """Close sqlite connecton and remove temprorary db
        """
        self._conn.close()
        self.delete_temp_db()


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='preprocess tsv file')
    
    parser.add_argument('--input', required=False,
                        metavar="/path/to/input/file",
                        help='Path to file to preprocess',
                        default=DEFAULT_INPUT_FILE_PATH)
    parser.add_argument('--output', required=False,
                        metavar="/path/to/output/file",
                        help="Path to output file",
                        default=DEFAULT_OUTPUT_FILE_PATH)
    parser.add_argument('--chunksize', required=False,
                        metavar="chunksize",
                        help="chunksize",
                        default=DEFAULT_CHUNK_SIZE)
   
    args = parser.parse_args()

    print("Input file path: ", args.input)
    print("Output file path: ", args.output)
    print("chunksize: ", args.chunksize)

    class MyConfig(Config):
        INPUT_FILE_PATH = args.input
        OUTPUT_FILE_PATH = args.output
        CHUNK_SIZE = int(args.chunksize)

    my_config = MyConfig()

    process = Process(my_config)
    process.process_data()

    print('Preprocess has finished')

 

