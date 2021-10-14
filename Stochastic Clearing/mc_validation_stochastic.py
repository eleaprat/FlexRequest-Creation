"""
Monte Carlo validation of the outputs of the FlexRequest creation model for a number of out-of-sample scenarios.

06.08.2021 - I.D.
"""

import numpy as np
import pandas as pd

def mc_out_validation(forecast_err, results, network_data, p_inc_mat, v_inc_mat, uncertainty_inc_mat, nb_scn, nb_t, pf, display_results, case_name, offer_case):
    """

    :param uncert_samples: Input samples
    :param flex_setpoints: Dictionary with results from the FlexRequest creation: positive (flex_p) and
                            negative (flex_n) requests
    :return:
    S_line: line loading in percentage
    V: voltage level in p.u.
    """

    if display_results == True:
        print('----------------------Monte Carlo out-of-sample validation----------------------')
        
    
    node_df = network_data['bus_data']
    nb_bus = node_df.shape[0]
    branch_df = network_data['branch_data']
    nb_lines = branch_df.shape[0]
    baseMVA = network_data['base_MVA']
    # N = list(node_df['Bus'])
    # B = [(branch_df.loc[i, 'From'], branch_df.loc[i, 'To']) for i in branch_df.index]
    T = range(1,nb_t+1)
    
    epsilon = 0.000001 # Tolerance
    
    factor_Q = np.sqrt((1-pf ** 2)/pf ** 2)
    R = np.diag(np.array(branch_df['R'].to_list()))
    X = np.diag(np.array(branch_df['X'].to_list()))
    
    P_act = {}
    P_off_p = {}
    P_off_n = {}
    P_curt={}
    P_NS={}
    alpha = {}
    alpha_act = {}
    alpha_curt = {}
    alpha_NS = {}
    V_sq = {}
    Vmin_sq = {}
    Vmax_sq = {}
    P = {}
    Smax_sq = {}
    for t in T:
        for i in node_df.index:
            P_act[node_df.loc[i, 'Bus'],t] = results['P_act'][i+(t-1)*nb_bus]/baseMVA
            P_off_p[node_df.loc[i, 'Bus'],t] = results['P_off_p'][i+(t-1)*nb_bus]/baseMVA
            P_off_n[node_df.loc[i, 'Bus'],t] = results['P_off_n'][i+(t-1)*nb_bus]/baseMVA
            P_curt[node_df.loc[i, 'Bus'],t] = results['P_curt'][i+(t-1)*nb_bus]/baseMVA
            P_NS[node_df.loc[i, 'Bus'],t] = results['P_NS'][i+(t-1)*nb_bus]/baseMVA
            alpha[node_df.loc[i, 'Bus'],t] = results['alpha'][i+(t-1)*nb_bus]
            alpha_act[node_df.loc[i, 'Bus'],t] = results['alpha_act'][i+(t-1)*nb_bus]
            alpha_curt[node_df.loc[i, 'Bus'],t] = results['alpha_curt'][i+(t-1)*nb_bus]
            alpha_NS[node_df.loc[i, 'Bus'],t] = results['alpha_NS'][i+(t-1)*nb_bus]
            V_sq[node_df.loc[i, 'Bus'],t] = results['V_sq'][i+(t-1)*nb_bus]
            Vmin_sq[node_df.loc[i, 'Bus']] = node_df.loc[i, 'Vmin'] ** 2
            Vmax_sq[node_df.loc[i, 'Bus']] = node_df.loc[i, 'Vmax'] ** 2
        for b in branch_df.index:
            P[branch_df.loc[b, 'From'], branch_df.loc[b, 'To'],t] = results['P'][b+(t-1)*nb_lines]/baseMVA
            Smax_sq[branch_df.loc[b, 'From'], branch_df.loc[b, 'To']] = (branch_df.loc[b, 'Lim']/baseMVA)**2
    
    # For each CC, count number of violations
    cc_1 = {}
    cc_2 = {}
    cc_3 = {}
    cc_4 = {}
    cc_5 = {}
    cc_6 = {}
    cc_7 = {}
    for t in T:
        for bus_i in node_df.index:
            i = node_df.loc[bus_i,'Bus']
            cc_1[i,t] = 0
            cc_2[i,t] = 0
            cc_3[i,t] = 0
            cc_4[i,t] = 0
            cc_5[i,t] = 0
            cc_6[i,t] = 0
        for line_i in branch_df.index:
            i = branch_df.loc[line_i, 'From']
            j = branch_df.loc[line_i, 'To']
            cc_7[i,j,t] = 0
            
    for s in range(1,nb_scn+1):
        P_act_s = {}
        P_curt_s = {}
        P_NS_s = {}
        V_sq_s = {}
        P_s = {}
        for t in T:
            error = np.array([f/baseMVA for f in forecast_err[s,t]])
            error_tot = sum(error)
            alpha_vec = np.array([alpha[i,t] for i in node_df['Bus']])
            for bus_i in node_df.index:
                i = node_df.loc[bus_i,'Bus']
                P_act_s[i,t] = P_act[i,t] + alpha_act[i,t] * error_tot
                # cc_1
                cc_1_prev = cc_1[i,t]
                if P_act_s[i,t] < - (P_off_n[i, t] + epsilon):
                    cc_1_prev+=1
                    cc_1[i,t] = cc_1_prev
                # cc_2
                cc_2_prev = cc_2[i,t]
                if P_act_s[i,t] > (P_off_p[i, t] + epsilon):
                    cc_2_prev+=1
                    cc_2[i,t] = cc_2_prev
                
                P_curt_s[i,t] = P_curt[i,t] + alpha_curt[i,t] * error_tot
                # cc_3
                cc_3_prev = cc_3[i,t]
                if P_curt_s[i,t] < - epsilon:
                    cc_3_prev+=1
                    cc_3[i,t] = cc_3_prev
                
                P_NS_s[i,t] = P_NS[i,t] + alpha_NS[i,t] * error_tot
                # cc_4
                cc_4_prev = cc_4[i,t]
                if P_NS_s[i,t] < - epsilon:
                    cc_4_prev+=1
                    cc_4[i,t] = cc_4_prev
                
                V_sq_s[i,t] = V_sq[i,t] - 2*v_inc_mat[bus_i,:] @ ((R + factor_Q**2 * X) @ p_inc_mat @ (uncertainty_inc_mat @ error - error_tot * alpha_vec))
                # cc_5
                cc_5_prev = cc_5[i,t]
                if V_sq_s[i,t] < (Vmin_sq[i] - epsilon):
                    cc_5_prev+=1
                    cc_5[i,t] = cc_5_prev
                # cc_6
                cc_6_prev = cc_6[i,t]
                if V_sq_s[i,t] > (Vmax_sq[i] + epsilon):
                    cc_6_prev+=1
                    cc_6[i,t] = cc_6_prev
            for line_i in branch_df.index:
                i = branch_df.loc[line_i, 'From']
                j = branch_df.loc[line_i, 'To']
                P_s[i,j,t] = P[i,j,t] + p_inc_mat[line_i, :] @ (uncertainty_inc_mat @ error - error_tot * alpha_vec)
                # cc_7
                cc_7_prev = cc_7[i,j,t]
                if P_s[i,j,t]**2 * (1 + factor_Q**2) > (Smax_sq[i,j] + epsilon):
                    cc_7_prev+=1
                    cc_7[i,t] = cc_7_prev
    
    CC_df = pd.DataFrame(columns=['Constraint','ID','Value'])
    
    for t in T:
        for bus_i in node_df.index:
            i = node_df.loc[bus_i,'Bus']
            cc_1_prev = cc_1[i,t]
            cc_1[i,t] = cc_1_prev / nb_scn
            cc_2_prev = cc_2[i,t]
            cc_2[i,t] = cc_2_prev / nb_scn
            cc_3_prev = cc_3[i,t]
            cc_3[i,t] = cc_3_prev / nb_scn
            cc_4_prev = cc_4[i,t]
            cc_4[i,t] = cc_4_prev / nb_scn
            cc_5_prev = cc_5[i,t]
            cc_5[i,t] = cc_5_prev / nb_scn
            cc_6_prev = cc_6[i,t]
            cc_6[i,t] = cc_6_prev / nb_scn
            
            if cc_1_prev > 0:
                new_row = {'Constraint':'Chance constraint 1 (P_act >= - P_off_down)','ID':'({},{})'.format(i,t),'Value':'{}%'.format(round(cc_1_prev*100 / nb_scn,1))}
                CC_df = CC_df.append(pd.Series(new_row), ignore_index=True)
                if display_results == True:
                    print('Chance constraint 1 ({},{}):{}% '.format(i,t,round(cc_1_prev*100 / nb_scn,1)))
            if cc_2_prev > 0:
                new_row = {'Constraint':'Chance constraint 2 (P_act <= P_off_up)','ID':'({},{})'.format(i,t),'Value':'{}%'.format(round(cc_2_prev*100 / nb_scn,1))}
                CC_df = CC_df.append(pd.Series(new_row), ignore_index=True)
                if display_results == True:
                    print('Chance constraint 2 ({},{}):{}% '.format(i,t,round(cc_2_prev*100 / nb_scn,1)))
            if cc_3_prev > 0:
                new_row = {'Constraint':'Chance constraint 3 (P_NS >= 0)','ID':'({},{})'.format(i,t),'Value':'{}%'.format(round(cc_3_prev*100 / nb_scn,1))}
                CC_df = CC_df.append(pd.Series(new_row), ignore_index=True)
                if display_results == True:
                    print('Chance constraint 3 ({},{}):{}% '.format(i,t,round(cc_3_prev*100 / nb_scn,1)))
            if cc_4_prev > 0:
                new_row = {'Constraint':'Chance constraint 4 (P_curt >= 0)','ID':'({},{})'.format(i,t),'Value':'{}%'.format(round(cc_4_prev*100 / nb_scn,1))}
                CC_df = CC_df.append(pd.Series(new_row), ignore_index=True)
                if display_results == True:
                    print('Chance constraint 4 ({},{}):{}% '.format(i,t,round(cc_4_prev*100 / nb_scn,1)))
            if cc_5_prev > 0:
                new_row = {'Constraint':'Chance constraint 5 (V >= V_min)','ID':'({},{})'.format(i,t),'Value':'{}%'.format(round(cc_5_prev*100 / nb_scn,1))}
                CC_df = CC_df.append(pd.Series(new_row), ignore_index=True)
                if display_results == True:
                    print('Chance constraint 5 ({},{}):{}% '.format(i,t,round(cc_5_prev*100 / nb_scn,1)))
            if cc_6_prev > 0:
                new_row = {'Constraint':'Chance constraint 6 (V <= V_max)','ID':'({},{})'.format(i,t),'Value':'{}%'.format(round(cc_6_prev*100 / nb_scn,1))}
                CC_df = CC_df.append(pd.Series(new_row), ignore_index=True)
                if display_results == True:
                    print('Chance constraint 6 ({},{}):{}% '.format(i,t,round(cc_6_prev*100 / nb_scn,1)))
        for line_i in branch_df.index:
            i = branch_df.loc[line_i, 'From']
            j = branch_df.loc[line_i, 'To']
            cc_7_prev = cc_7[i,j,t]
            cc_7[i,j,t] = cc_7_prev / nb_scn
            if cc_7_prev > 0:
                new_row = {'Constraint':'Chance constraint 7 (Line limit)','ID':'({},{})'.format(i,t),'Value':'{}%'.format(round(cc_7_prev*100 / nb_scn,1))}
                CC_df = CC_df.append(pd.Series(new_row), ignore_index=True)
                if display_results == True:
                 print('Chance constraint 7 ({},{},{}):{}% '.format(i,j,t,round(cc_7_prev*100 / nb_scn,1)))

    CC_df.to_csv('{}_{}_CC_results_SC.csv'.format(case_name, offer_case))
    
    cc = [cc_1, cc_2, cc_3, cc_4, cc_5, cc_6, cc_7]
    
    return cc