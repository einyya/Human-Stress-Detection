from HumanDataExtraction import HumanDataExtraction
from HumanDataPebl import HumanDataPebl
from AnalysisData import AnalysisData


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #DataPath = r'C:\Users\97254\OneDrive\Desktop\Human Bio Signals Analysis'
    DataPath = r'C:\Users\e3bom\Desktop\Human Bio Signals Analysis'
    ex_col=['Class', 'Test_Type', 'Level', 'Accuracy','RT', 'Stress','Stress_N','Stress_D', 'Fatigue',
            'Fatigue_N','Fatigue_D','ID','Efficiency_S','Efficiency_F', 'Involvement_S', 'Involvement_F',
            'SAT', 'IES','Group', 'Time','Window','Overlap']
    Prediction_Targets=['Involvement_S','Accuracy','SAT', 'IES']
    Prediction_Targets1=['Stress','Stress_N','Stress_D','Fatigue','Fatigue_N','Fatigue_D','Efficiency_S','Efficiency_F', 'Involvement_F',]
    miss=['Involvement_F_Base']
    Make_Trigger=False
    Make_DataSet=False
    Analysis_DataSet=True

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

        hde = HumanDataExtraction(DataPath,ex_col)
        if PreProcessing:
            hde.Check_MedianFilter(ID=None,rangeID=False)
            hde.Check_MinMaxVlaue(ID=None,rangeID=False)

        if Dataset:
            hde.CleanData(ID=None,rangeID=True)
            hde.CreateDataset(ID=None,rangeID=True)
            hde.MissingData(ID=None, rangeID=False)
            hde.Create_Delta()
            hde.CorrelationMatrixAndReduce()

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
        ad = AnalysisData(DataPath,ex_col,Prediction_Targets)
        # ad.StatisticalTest()
        # ad.GroupDiffPlot()
        ad.ML_models_Prediction()
        # ad.ML_models_Classification(n_repeats=9, no_breath_data=True, clases_3=True)
        # ad.ML_models_Classification(n_repeats=9, no_breath_data=True, clases_3=False)
        # ad.ML_models_Classification(n_repeats=9, no_breath_data=False, clases_3=True)
        # ad.ML_models_Classification(n_repeats=9, no_breath_data=False, clases_3=False)



