import neurokit2 as nk
rsp = nk.rsp_simulate(duration=90, respiratory_rate=15)

rsp, info = nk.rsp_process(rsp)

nk.rsp_rrv(rsp, show=True)