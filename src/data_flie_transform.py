from abc import ABC, abstractmethod
import os
import numpy as np

class DataFileTransformFun(ABC):
    @abstractmethod
    def __call__(self):
        pass

class FileToData(DataFileTransformFun):
    @abstractmethod
    def __call__(self):
        pass

class DataToFile(DataFileTransformFun):
    @abstractmethod
    def __call__(self):
        pass

class DataToData(DataFileTransformFun):
    @abstractmethod
    def __call__(self):
        pass

# mat file -> npdata
class MatFileToNpdata(FileToData):
    def __call__(self, file_folder, file_name, fea_name):
        import h5py
        file = h5py.File(os.path.join(file_folder, file_name))
        npdata = file[fea_name][:]
        return npdata

# npz file -> npdata 
class NpzFileToNpdata(FileToData):
    def __call__(self, file_folder, file_name, fea_name):
        npdata_dict = np.load(os.path.join(file_folder, file_name))
        npdata = npdata_dict[fea_name]
        return npdata

# npy file -> npdata
class NpyFileToNpdata(FileToData):
    def __call__(self, file_folder, file_name):
        npdata = np.load(os.path.join(file_folder, file_name))
        return npdata

# data meta file -> npdata
class DataMetaFileToNpdata(FileToData):
    def __call__(self, file_folder, file_name):
        from MITgcmutils import mds
        npdata = mds.rdmds(os.path.join(file_folder, file_name))
        return npdata

# t ids data meta file -> npdata
class TIidsDataMetaFileToNpdata(FileToData):
    def __call__(self, file_folder, file_prefix_name, t_ids):
        from MITgcmutils import mds
        npdata_list = []
        for id in t_ids:
            data = mds.rdmds(os.path.join(file_folder, file_prefix_name), id)
            npdata_list.append(data)
        npdata = np.stack(npdata_list, axis=0)
        return npdata

# p, b, s npdata -> rho npdata
class StateParaNpdataDictToRhoNpdata(DataToData):
    def __call__(self, npdata_p, npdata_b, npdata_s):
        pass

# npz file -> npdata dict
class NpzFileToNpdataDict(FileToData):
    def __call__(self, file_folder, file_name):
        npdata_dict = np.load(os.path.join(file_folder, file_name))
        npdata_dict = {key: npdata_dict[key] for key in npdata_dict.files}

        return npdata_dict

# npdata dict -> npz file
class NpdataDictToNpzFile(DataToFile):
    def __call__(self, npdata_dict, file_folder, file_name):
        # np.savez(os.path.join(file_path, file_name), s = npdata_dict["s"], b = npdata_dict["b"])
        np.savez(os.path.join(file_folder, file_name), **npdata_dict)

# npdata -> npy file
class NpdataToNpyFile(DataToFile):
    def __call__(self, npdata, file_folder, file_name):
        np.save(os.path.join(file_folder, file_name), npdata)

# nc file -> npdata
class NcFileToNpdata(FileToData):
    def __call__(self, file_folder, file_name, fea_name):
        import netCDF4 as nc
        with nc.Dataset(os.path.join(file_folder, file_name), "r") as ncdata:
            npdata = ncdata.variables[fea_name][:]
        return npdata

# multi process nc file -> npdata
class MultiXNcFileToNpdata(FileToData):
    def __call__(self, file_folder, file_prefix_name, fea_name):
        import netCDF4 as nc
        import glob

        file_path_name_list = glob.glob(os.path.join(file_folder, f"{file_prefix_name}*"))
        file_path_name_list = sorted(file_path_name_list)
        npdata_list = []
        for file_path_name in file_path_name_list:
            with nc.Dataset(file_path_name, "r") as ncdata:
                npdata = ncdata.variables[fea_name][:]
                npdata_list.append(npdata)
        npdata = np.concatenate(npdata_list, axis=-1)

        return npdata
    
# npdata -> mat file
class NpdataToMatFile(DataToFile):
    def __call__(self, npdata, file_folder, file_name, fea_name):
        from scipy.io import savemat
        savemat(os.path.join(file_folder, file_name), {fea_name: npdata})

def test():
    # super param
    root_folder = "."
    data_folder = os.path.join(root_folder, "data")

    # mat file -> npdata
    transform_fun = MatFileToNpdata()
    npdata_u = transform_fun(file_folder=os.path.join(data_folder, "mat_file"), file_name="U_velocity.mat", fea_name="uvel")

    # npz file -> npdata 
    transform_fun = NpzFileToNpdata()
    npdata_rho = transform_fun(file_folder=os.path.join(data_folder, "npz_file"), file_name="suitable_data.npz", fea_name="rho")

    # npdata -> npy file
    transform_fun = NpdataToNpyFile()
    transform_fun(npdata=npdata_rho, file_folder=os.path.join(data_folder, "npy_file"), file_name="rho.npy")

    # npy file -> npdata
    transform_fun = NpyFileToNpdata()
    npdata = transform_fun(file_folder=os.path.join(data_folder, "npy_file"), file_name="rho.npy")

    # data meta file -> npdata
    transform_fun = DataMetaFileToNpdata()
    npdata = transform_fun(file_folder=os.path.join(data_folder, "data_meta"), file_name="SALT.0000048350.001.001")

    # t indexs data meta file -> npdata
    transform_fun = TIidsDataMetaFileToNpdata()
    t_ids = range(1000, 20000+1, 1000)
    npdata_s = transform_fun(file_folder=os.path.join(data_folder, "data_meta"), file_prefix_name="SALT" , t_ids=t_ids)
    npdata_b = transform_fun(file_folder=os.path.join(data_folder, "data_meta"), file_prefix_name="THETA", t_ids=t_ids)

    # npdata dict -> npz file
    npdata_dict = {"s": npdata_s, "b": npdata_b}
    transform_fun = NpdataDictToNpzFile()
    transform_fun(npdata_dict=npdata_dict, file_folder=os.path.join(data_folder, "npz_file"), file_name="data.npz")

    # npz file -> npdata dict
    transform_fun = NpzFileToNpdataDict()
    npdata_dict = transform_fun(file_folder=os.path.join(data_folder, "npz_file"), file_name="data.npz")

    # nc file -> npdata
    transform_fun = NcFileToNpdata()
    npdata_u = transform_fun(file_folder=os.path.join(data_folder, "nc_file"), file_name="stateU.0000000000.t011.nc", fea_name="U")
    print(f"shape: {npdata_u.shape}")

    # out: shape: (433, 100, 1, 51)

    # multi process nc file -> npdata
    transform_fun = MultiXNcFileToNpdata()
    npdata_multi_u = transform_fun(file_folder=os.path.join(data_folder, "nc_file"), file_prefix_name="stateU", fea_name="U")
    
    # npdata -> mat file
    transform_fun = NpdataToMatFile()
    transform_fun(npdata=npdata_multi_u, file_folder=os.path.join(data_folder, "mat_file"), file_name="multi_u.mat", fea_name="multi_u")

def main():
    test()

if __name__ == "__main__":
    main()
