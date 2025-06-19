from HumanDataExtraction import HumanDataExtraction
from HumanDataPebl import HumanDataPebl
from AnalysisData import AnalysisData


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #DataPath = r'C:\Users\97254\OneDrive\Desktop\Human Bio Signals Analysis'
    DataPath = r'C:\Users\e3bom\Desktop\Human Bio Signals Analysis'
    Make_Trigger=False
    Make_DataSet=True
    Analysis_DataSet=False

    #________________________________________________________Make Trigger Table_______________________________________________________
    if Make_Trigger:
        hdp = HumanDataPebl(DataPath)
        hdp.CreateDataset_PerformanceScore(ID=None,rangeID=True)
        hdp.Make_Trigger_Table(ID=None,rangeID=True)
        hdp.CreateDataset_StressScore(ID=None,rangeID=True)
    #________________________________________________________SortData_______________________________________________________
    if Make_DataSet:

        PreProcessing = False
        Dataset=True
        Combine = False
        HRV = False
        RSP = False
        EDA = False

        hde = HumanDataExtraction(DataPath)
        if PreProcessing:
            hde.Check_MedianFilter(ID=None,rangeID=False)
            hde.Check_MinMaxVlaue(ID=None,rangeID=False)

        if Dataset:
            # hde.CleanData(ID=65,rangeID=True)
            hde.CreateDataset(ID=None,rangeID=True)
            # hde.MissingData(ID=None, rangeID=False)

        if Combine:
            hde.AX_plot_signals_VAS(ID=52,rangeID=True,Signals_plot=True,Cor_plot=False)

        if RSP:
            hde.RSP_Parts(ID=None, Group=None)

        if EDA:
            hde.AX_plot_3_part_HRV(ID = [19,20,21])
            hde.AX_plot_3in1norm_EDA(ID=[19, 20, 21])
            hde.AX_plot_3in1_EDA(ID=[19, 20, 21])
            hde.AX_plot_3_part_EDA(ID=[19, 20, 21])
        if HRV:
            hde.HRV_Window_Feature(ID=None,rangeID=True)
            hde.HRV_Window_Feature_all()
            hde.HRV_Window_2Features(ID = 27, Group = 'breath')


    #________________________________________________________SortData_______________________________________________________
    if Analysis_DataSet:
        ad = AnalysisData(DataPath)
        ad.ML_models_Classification()
        # ad.ML_models_Prediction()
        ad.GroupDiff()
