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
        hdp.Make_Trigger_Table(ID=42,rangeID=True)
    #________________________________________________________SortData_______________________________________________________
    if Make_DataSet:

        PreProcessing = False
        Dataset=False
        Combine = True
        HRV = False
        RSP = False
        EDA = False

        hde = HumanDataExtraction(DataPath)
        if PreProcessing:
            hde.Check_MedianFilter(ID=None,rangeID=False)

        if Dataset:
            # hde.CleanData(ID=42,rangeID=True)
            hde.CreateDataset(ID=42,rangeID=True)

        if Combine:
            hde.AX_plot_signals_VAS(ID=41,rangeID=True)

        if RSP:
            hde.RSP_Parts(ID=None, Group=None)

        if EDA:
            hde.AX_plot_3_part_HRV(ID = [19,20,21])
            hde.AX_plot_3in1norm_EDA(ID=[19, 20, 21])
            hde.AX_plot_3in1_EDA(ID=[19, 20, 21])
            hde.AX_plot_3_part_EDA(ID=[19, 20, 21])
        if HRV:
            hde.HRV_Window_Feature(ID=None)
            hde.HRV_Window_Feature_all()
            hde.HRV_Window_2Features(ID = 27, Group = 'breath')


    #________________________________________________________SortData_______________________________________________________
    if Analysis_DataSet:
        ad = AnalysisData(DataPath)
        ad.ML_models_all()
