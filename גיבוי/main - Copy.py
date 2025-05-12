from HumanDataExtraction import HumanDataExtraction
from HumanDataPebl import HumanDataPebl
from AnalysisData import AnalysisData


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DataPath = r'C:\Users\e3bom\Desktop\Human Bio Signals Analysis'
    Make_Trigger=False
    Make_DataSet=True
    Analysis_DataSet=True

    #________________________________________________________Make Trigger Table_______________________________________________________
    if Make_Trigger:
        hdp = HumanDataPebl(DataPath)
        hdp.Make_Trigger_Table(ID=None,rangeID=False)
    #________________________________________________________SortData_______________________________________________________
    if Make_DataSet:

        PreProcessing = False
        Dataset=True
        Combine = True
        HRV = True
        RSP = False
        EDA = False

        hde = HumanDataExtraction(DataPath)
        if PreProcessing:
            hde.Check_MedianFilter(ID=None,rangeID=False)

        if Dataset:
            hde.CleanData(ID=None,rangeID=False)
            hde.CreateDataset(ID=None,rangeID=False)

        if Combine:
            hde.AX_plot_signals_VAS(ID=None,rangeID=False)

        if RSP:
            hde.RSP_Parts(ID=None, Group=None)

        if EDA:
            hde.AX_plot_3_part_HRV(ID = [19,20,21])
            hde.AX_plot_3in1norm_EDA(ID=[19, 20, 21])
            hde.AX_plot_3in1_EDA(ID=[19, 20, 21])
            hde.AX_plot_3_part_EDA(ID=[19, 20, 21])
        if HRV:
            hde.HRV_Window_Feature(ID=None, Group=None)
            hde.HRV_Window_Feature_all()
            hde.HRV_Window_2Features(ID = 27, Group = 'breath')
            hde.AX_plot_3_part_HRV()

    #________________________________________________________SortData_______________________________________________________
    if Analysis_DataSet:
        ad = AnalysisData(DataPath)
        ad.ML_models_all()
    # # Utilities.save_dataframe(hde.sorted_DATA, dataframe_path, f'{Analysis_Type}_Sorted_DATA')



