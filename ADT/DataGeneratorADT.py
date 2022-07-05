from nibabel.filebasedimages import ImageFileError
import os

class Empty(Exception):
    pass


class DataGenerator:
    """
    The class allows referential storage of data and batched data generation

    Attributes
    ----------
    _path: path to the data unit files (DUFs)
    _size: number of stored datasets
    _IDs: dictionary mapping IDs to DUF files
    _DUF_type: type of DUf (currently supported: 'csv';'csv_w_header')
    _DUF_columns: data column names

    Methods
    -------
    len: returns the number of stored datasets ( = attribute _size)
    isEmpty(): returns True if len(object)==0
    setPath(path): sets the _path attribute to path
    getPath(): returns the _path attribute
    setftype(ftype): sets the _ftype attribute to ftype
    loadDUF(filenameRegex, path='', ID_cut=[0, -1], ftype='csv'): loads DUFs from path using filenameRegex, extracts ID
        using ID_cut on the filename and store corresponding ID (=key) and filename (=value) in _IDs attribute
    split(ratio=0.8, randomly=True): splits the object to two DataGenerator objects with a given data-ratio
    generate(IDList, outputFormat='np.ndarray'): generate batched data of the listed IDs in IDList
    IDmatch(other): applies an inner join on the _IDs and other DataGenerator object _IDs
    """

    class IDMismatch(Exception):
        pass

    def __init__(self):
        self._path = ''
        self._size = 0
        self._IDs = {}  # dictionary with key = ID and value = DUF filename
        self._DUF_type = None
        self._DUF_columns = None

    def __len__(self):
        """
        :return: the number of instances referenced from the DataGenerator object
        """

        return self._size

    def isEmpty(self):
        """
        :return: True if there are no instances referenced from the DataGenerator, False otherwise
        """

        return self._size == 0

    def setPath(self, path):
        """
        set the DataGenerator (DUF-) path attribute to path

        :param path: string indicating the path to DUF
        :return: True if the path exists in the system and False otherwise
        """
        import os
        if type(path) != str:
            raise TypeError('Path variable must be a string of a valid path.')
        print('DUF path:', path)
        self._path = path
        if os.path.exists(path):
            return True
        else:
            raise Warning('The given path does not exist in the system:', path, 'Current working directory:',
                          os.getcwd())
            return False

    def getPath(self):
        """
        :return: The function returns the (DUF) path attribute of the DataGenerator object
        """

        return self._path

    def setftype(self, ftype):
        """
        Set the DUF data type to ftype

        :param ftype: string specifying the DUF data type. Currently supported options: 'csv';'csv_w_header'
        :return: True when done
        """

        if ftype != 'csv' and ftype != 'csv_w_header' and ftype != 'nii':
            raise ValueError('The DataGenerator supports only the following ftype: \'csv\', \'csv_w_header\', \'npy\', \'nii\'')
        self._DUF_type = ftype
        return True

    def loadDUF(self, filenameRegex, path='', ID_cut=[0, -1], ftype='csv'):
        """
        Load data unit files (DUFs) to the DataGenerator object

        The DUFs are stored in the DataGenerator _IDs attribute as a dictionary mapping each ID the corresponding file

        :param filenameRegex: a regex string representing describing the DUFs filenames to load
        :param path: the path to the folder containing the DUF files. If not given, the DG will attempt to load the
            files from the path attribute
        :param ID_cut: An ID will be extracted from each loaded filename according to ID=filename[ID_cut[0]:ID_cut[1]]
            if not given, the full filename will be stored as an ID
        :param ftype: string indicating the DUF type. Supported types are: 'csv'; 'csv_w_header' ;'npy'; 'nii'
        :return: The number of loaded DUFs (0 if none was loaded): integer
        """

        import os, glob
        # check input variables for type and values
        # Make sure the path (or _path attribute if path==None) exists in the system
        if not path:
            if not self._path:
                raise Empty('The object does not have a stored path. Please re-run with a path string')
        else:
            if type(path) != str:
                raise TypeError('path variable must be a string of a valid path')
            elif not os.path.exists(path):
                raise Warning('The given path = ', path, ' does not exist in the system')
            else:
                self._path = path
        if type(filenameRegex) != str:
            raise TypeError('filenameRegexp must be a string')
        if len(ID_cut) != 2:
            raise ValueError('ID_cut must be a list of two integers')
        if type(ID_cut[0]) != int or type(ID_cut[1]) != int:
            raise TypeError('ID_cut must be a list of two integers')
        if ID_cut[1] != -1 and ID_cut[0] >= ID_cut[1]:
            raise ValueError('ID_cut[0] must be larger than ID_cut[1] in order to make a valid cut')
        if ftype != 'csv' and ftype != 'csv_w_header' and ftype != 'npy' and ftype != 'nii' and ftype != 'nii.gz':
            raise TypeError('The ftype ', ftype, ' is not a supported DUF type')

        self._DUF_type = ftype
        current_directory = os.getcwd()
        os.chdir(self._path)
        files = glob.glob(filenameRegex)
        if len(files) == 0:
            return 0
        loaded = 0
        for f in files:
            ID = f[ID_cut[0]:ID_cut[1]]
            self._IDs[ID] = os.path.join(self._path, f)
            loaded += 1
        self._size = len(self._IDs)
        os.chdir(current_directory)
        return loaded

    def getIDs(self):
        """
        :return: The function returns a list of the IDs (keys) stored in the DataGenerator
        """

        return list(self._IDs.keys())

    def split(self, selected_IDs=None, ratio=0.8, randomly=True):
        """
        Split the DataGenerator object to two new DataGenerator objects with data split according to ratio

        :param selected_IDs: optionally a list of IDs to be seperated -> used when splits are determined externally
        :param ratio: float in the range (0,1) representing the data split ratio
        :param randomly: if True the IDs are randomly shuffled before the split
        :return: DG1 with (ratio) of the data and DG2 with (1-ratio) of the data
        """

        import copy
        # check input
        if not isinstance(ratio, float) or ratio > 1 or ratio < 0:
            raise ValueError('ratio must be a float value between 0 and 1')
        ID_list = self.getIDs()
        if randomly:
            import random
            #random.seed(5)
            random.seed(7)
            random.shuffle(ID_list)
        N = len(ID_list)
        IDs_2 = ID_list[int(N * ratio):]

        DG1 = copy.deepcopy(self)
        DG2 = DataGenerator()
        for ID in IDs_2:
            DG2._IDs[ID] = DG1._IDs.pop(ID)
            DG2._size += 1
            DG1._size -= 1
        DG2._DUF_type = DG1._DUF_type
        DG2._DUF_columns = DG1._DUF_columns

        return DG1, DG2

    def generate(self, IDList, outputFormat='np.ndarray'):
        """
        Generate a unified data-matrix of the listed IDs. test!

        :param IDList: list of ID strings
        :param outputFormat: the output data format. Supported options are: 'np.ndarray' ; 'pd.df'
        :return: The unified data-matrix with the first dimension as the number of observations (i.e. == len(IDList))
        """

        import csv
        import numpy as np
        import pandas as pd

        # check input
        if len(IDList) < 1:
            raise ValueError('To generate batched data you must provide a list of IDs via IDList')
        if outputFormat != 'np.ndarray' and outputFormat != 'pd.df':
            raise ValueError('The DataGenerator supports only \'np.ndarry\' and \'pd.df\' output formats.Set outputFormat to one of those values')
        if outputFormat == 'pd.df' and self._DUF_type == 'nii':
            raise ValueError('For ftype = \'nii\' only \'np.ndarray\' is a supported outputFormat.')
        if not self._IDs or not self._DUF_type:
            raise Empty('The DG is empty. Use the loadDUF method before you generate data')

        output = []
        for ID in IDList:
            if ID not in self._IDs:
                raise self.IDMismatch('The listed ID', ID, 'does not exist in DG', self)
            file = self._IDs[ID]
            if self._DUF_type == 'npy':
                output.append(np.load(self._IDs[ID]))
            elif self._DUF_type == 'csv' or self._DUF_type == 'csv_w_header':
                with open(file, 'r') as csvFile:
                    if self._DUF_type == 'csv_w_header':
                        #header = next(reader)
                        header = csvFile.readline().strip('\n').split(',')[0:]
                        if not self._DUF_columns:
                            self._DUF_columns = header
                        if header != self._DUF_columns:
                            print('The DUF header is not consistent with the DG setting.')
                            print('The DUF header was:', self._DUF_columns, 'and is now set to:', str(header))
                            self._DUF_columns = header
                    reader = csv.reader(csvFile, quoting=csv.QUOTE_NONNUMERIC)
                    list_reader = list(reader)
                    output.append(list_reader[0])  # this was the buggy original code
            elif (self._DUF_type == 'nii' or self._DUF_type == 'nii.gz'):
                import nibabel as nib
                from nibabel.filebasedimages import ImageFileError
                try:
                    output.append(nib.load(file).get_fdata())
                except ImageFileError:
                    output.append(nib.load(file + '.gz').get_fdata())

        output = np.asarray(output)
        try:
            output = np.squeeze(output, 1) # squeeze the redundant dimension
        except:
            pass
        # transform the merged data into the requested output type:
        if outputFormat == 'pd.df':
            if self._DUF_columns:
                output = pd.DataFrame(output, columns=self._DUF_columns)
            else:
                output = pd.DataFrame(output)
        return output

    def IDmatch(self, other):
        """
        Apply an inner-join on the (self) DataGenerator object and other DataGenerator object

        :param other: A DataGenerator object
        :return: A copy of the other DG after inner join
        """

        import copy

        # check input:
        if not type(other) == DataGenerator:
            raise TypeError('DG must be a DataGenerator object')
        if self.isEmpty() or other.isEmpty():
            raise ValueError('One of the DG objects is empty, applying a join will delete the other DG!')

        for ID in list(self._IDs.keys()):
            if ID not in other._IDs.keys():
                del self._IDs[ID]
                self._size -= 1

        other_copy = copy.deepcopy(other)
        for ID in list(other._IDs.keys()):
            if ID not in self._IDs.keys():
                del other_copy._IDs[ID]
                other_copy._size -= 1
        return other_copy

    def ID_split(self, ID_list):

        import copy
        self_copy = copy.deepcopy(self)

        for ID in list(self_copy._IDs.keys()):
            if ID not in ID_list:
                del self_copy._IDs[ID]
                self_copy._size -= 1

        return self_copy

## Unit test:
class UnitTestError(Exception):
    pass


if __name__ == '__main__':

    import os

    test_path = 'C:\\Users\\miche\\OneDrive\\Charite\\Projects\\MMOP\\Testing'
    test_DUF_path = 'C:\\Users\\miche\\OneDrive\\Charite\\Projects\\MMOP\\Testing\\test_files_path'
    IDs2Generate = ['001', '002']
    DG_filename_01 = 'DG_01.pkl'
    os.chdir(test_path)

    DG = DataGenerator()
    print('getPath of empty DG:', DG.getPath())
    DG.setPath(test_path)
    print('getPath after using setPath:', DG.getPath())
    filename = 'DG_test'

    print('--------------- isEmpty method -----------------')
    if DG.isEmpty():
        print('isEmpty() outputs True when empty')
    else:
        raise UnitTestError('The isEmpty function did not return True when given an empty DG')

    if len(DG) == 0:
        print('len method works')
    else:
        raise UnitTestError('__len__ method returned wrong output for an empty DG')

    # test the load method by creating a new DG and loading it from file
    DG2 = DataGenerator()
    print('--------------- getIDs method -----------------')
    DG.loadDUF('test_header_1D*', test_DUF_path, [-7, -4], 'csv_w_header')
    listed_IDs = DG.getIDs()
    if listed_IDs != ['001','002','003','004']:
        raise UnitTestError('The getIDs method is erroneous! Please debug!')
    else:
        print('getIDs method works!')

    # check out split method
    print('--------------- split method -----------------')
    DG_01, DG_02 = DG.split(0.5)
    print('The resulted DGs after (random) 0.5 split:')
    print('DG_01:', DG_01.getIDs())
    print('DG_02:', DG_02.getIDs())

    print('--------------- getPath method -----------------')
    print('Path loaded path via loadDUF:', DG.getPath())
    DG2.loadDUF('test_2D*', test_DUF_path, [-7, -4])

    print('--------------- IDmatch method -----------------')
    left = DG.IDmatch(DG2)
    if left == len(DG) == len(DG2):
        print('IDmatch seems to work well!')
    else:
        UnitTestError('the IDmatch did not work! It yielded ', left, 'while left object has len ', len(DG),
                      'and right object has len ', len(DG2))

    print('DG IDs:', DG.getIDs())
    print('This should be the aggregated data for 1D with header for IDs [001,002]:')
    print('--------------- generate method -----------------')
    print(DG.generate(IDs2Generate))
    print('This should be the aggregated data for 2D without header for IDs [001,002]:')
    print(DG2.generate(IDs2Generate))

    print('--------------- test generate method for pd.df output-----------')
    print('correct version is running .. ')
    DUF_path = 'C:\\Users\\miche\\OneDrive\\Charite\\Projects\\MMOP\\Data\\DUFs'
    DUF_X_regex = '1000plus_clinData_7cov_w_header_X*'
    DUF_y_regex = '1000plus_clinData_7cov_w_header_y*'
    ID_cut = [-7, -4]
    ftype = 'csv_w_header'
    DG = DataGenerator()
    DG.loadDUF(DUF_X_regex, DUF_path, ID_cut, ftype)
    sample_IDs = DG.getIDs()[:5]
    batch = DG.generate(sample_IDs, 'pd.df')
    print(batch)