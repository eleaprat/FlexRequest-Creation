"""
Linear DistFlow AC power flow according to the Baran and Wu formulation.
Chance-constraints for bus voltage and line flow power included.
"""
import numpy as np
from pyomo.kernel import *
from pyomo.util.infeasible import log_infeasible_constraints
from scipy.stats import norm
import pandas as pd

def cc_lin_dist_flow_model(case_name, network_data, offers_file, pf, pow_inc_mat, vlt_inc_mat, uncertainty_inc_mat, viol_prob, beta_factor, costs, display_results):
    
    #Extract data
    baseMVA = network_data['base_MVA']
    node_data = network_data['bus_data']
    branch_data = network_data['branch_data']
    wf_data = network_data['wf_data']
    set_point = np.array(network_data['setpoint'])
    covariance_matrix_squared = network_data['sq_cov_mtx']
    wind_pf = network_data['wind_all_pf']
    offers = pd.read_csv('{} Data/{}.csv'.format(case_name,offers_file))
    
    # Create dictionnaries to retrieve the lines
    line_id = {(branch_data.at[l,'From'],branch_data.at[l,'To']):l for l in branch_data.index}
    
    # Retrieve slack bus
    ref = node_data[node_data['type'] == 3]['Bus'].to_list()[0] # Return the id of the slack bus

    # Determine number of time intervals based on the load input matrix
    if len(set_point.shape) > 1:
        len_T = set_point.shape[1]
    else:
        len_T = 1

    model = block()
    model.dual = suffix(direction=suffix.IMPORT)

    # Sets
    model.T = range(1,len_T+1)
    model.N = node_data['Bus'] # nodes
    model.B = [(branch_data.loc[i, 'From'], branch_data.loc[i, 'To'])
                                            for i in branch_data.index]  # lines
    model.O = offers.index.tolist()
    
    # Nodes with windfarm and initial wind
    wind_bus = []
    wind_id = []
    wind_init = []
    for b in node_data['Bus'].index:
        if node_data['Bus'][b] in wf_data['Bus'].values:
            wind_bus.append(node_data['Bus'][b])
            wind_id.append(b)
            wind_init.append(wind_pf[wf_data.index[wf_data['Bus'] == node_data['Bus'][b]].tolist()[0]])
        else:
            wind_init.append(0)
    model.W = wind_bus
    
    #%% Parameters
    
    # Costs
    model.cost_off = parameter_dict()
    for o in model.O:
        model.cost_off[o] = parameter(offers.loc[o,'Price'])
    model.cost_act_up = parameter_dict()
    for i in node_data.index:
        model.cost_act_up[node_data.loc[i, 'Bus']] = parameter(costs['A+'][i])
    model.cost_act_down = parameter_dict()
    for i in node_data.index:
        model.cost_act_down[node_data.loc[i, 'Bus']] = parameter(costs['A-'][i])
    model.cost_NS = parameter_dict()
    for i in node_data.index:
        model.cost_NS[node_data.loc[i, 'Bus']] = parameter(costs['NS'][i])
    model.cost_curt = parameter_dict()
    for i in node_data.index:
        model.cost_curt[node_data.loc[i, 'Bus']] = parameter(costs['C'][i])
    model.cost_req_up = parameter_dict()
    for i in node_data.index:
        model.cost_req_up[node_data.loc[i, 'Bus']] = parameter(costs['R+'][i])
    model.cost_req_down = parameter_dict()
    for i in node_data.index:
        model.cost_req_down[node_data.loc[i, 'Bus']] = parameter(costs['R-'][i])
    
    # Nodes
    model.Vmin_sq = parameter_dict()
    for i in node_data.index:
        model.Vmin_sq[node_data.loc[i, 'Bus']] = parameter(node_data.loc[i, 'Vmin'] ** 2)
    model.Vmax_sq = parameter_dict()
    for i in node_data.index:
        model.Vmax_sq[node_data.loc[i, 'Bus']] = parameter(node_data.loc[i, 'Vmax'] ** 2)

    # Setpoint
    factor_Q = np.sqrt((1-pf ** 2)/pf ** 2)
    P_ref = -(sum(set_point)+sum(wind_init))
    
    model.P_init = parameter_dict()
    for i in node_data.index:
        for t in model.T:
            if node_data.loc[i, 'Bus'] == ref:
                model.P_init[ref, 1] = parameter(P_ref /baseMVA)
            else:
                model.P_init[node_data.loc[i, 'Bus'], t] = parameter(set_point[i] /baseMVA)
    model.Q_init = parameter_dict()
    for i in node_data.index:
        for t in model.T:
            if node_data.loc[i, 'Bus'] == ref:
                model.Q_init[ref, 1] = parameter(factor_Q * P_ref /baseMVA)
            else: 
                model.Q_init[node_data.loc[i, 'Bus'], t] = parameter(factor_Q * set_point[i] /baseMVA)

    model.P_wind = parameter_dict()
    for i in node_data.index:
        for t in model.T:
            model.P_wind[node_data.loc[i, 'Bus'], t] = parameter(wind_init[i] /baseMVA)
    model.Q_wind = parameter_dict()
    for i in node_data.index:
        for t in model.T:
            model.Q_wind[node_data.loc[i, 'Bus'], t] = parameter(factor_Q * wind_init[i] /baseMVA)
    
    # Offers
    model.P_off_max = parameter_dict()
    for o in model.O:
        model.P_off_max[o] = parameter(offers.loc[o, 'Quantity'])

    # Lines
    model.Smax = parameter_dict()
    for i in branch_data.index:
        model.Smax[branch_data.loc[i, 'From'], branch_data.loc[i, 'To']] = parameter(branch_data.loc[i, 'Lim'] /baseMVA)
    model.R = parameter_dict()
    for i in branch_data.index:
        model.R[branch_data.loc[i, 'From'], branch_data.loc[i, 'To']] = parameter(branch_data.loc[i, 'R'])
    model.X = parameter_dict()
    for i in branch_data.index:
        model.X[branch_data.loc[i, 'From'], branch_data.loc[i, 'To']] = parameter(branch_data.loc[i, 'X'])
    
    # Define all matrix as Pyomo parameters
    model.p_inc_mtx = parameter_dict()
    model.xi_inc_mtx = parameter_dict()
    model.cov_mtx_sq = parameter_dict()

    # Fill the matrix Pyomo parameters with values
    for (i, j) in model.B:
        l = line_id[(i, j)]
        m = 0
        for k in model.N:
            model.p_inc_mtx[i, j, k] = parameter(pow_inc_mat[l, m])
            m+=1

    m = 0
    for k in model.N:
        n = 0
        for w in model.W:
            model.xi_inc_mtx[k, w] = parameter(uncertainty_inc_mat[m,n])
            n+=1
        m+=1
    
    m = 0
    for w1 in model.W:
        n = 0
        for w2 in model.W:
            model.cov_mtx_sq[w1, w2] = parameter(covariance_matrix_squared[m,n])
            n+=1
        m+=1
    
    # Probabilities for chance constraints violation
    S_viol_prob = viol_prob['S']
    V_viol_prob = viol_prob['V']
    A_viol_prob = viol_prob['A']
    C_viol_prob = viol_prob['C']
    NS_viol_prob = viol_prob['NS']

    #%% Variables
    
    model.V_sq = variable_dict()
    for i in model.N:
        for t in model.T:
            model.V_sq[i,t] = variable(lb=0, value=1)
    model.P = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.P[i,j,t] = variable(value=0)
    model.Q = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.Q[i,j,t] = variable(value=0)

    model.P_off = variable_dict()
    for o in model.O:
        model.P_off[o] = variable(lb=0, ub=model.P_off_max[o], value=0)
    model.P_off_p = variable_dict()
    for i in model.N:
        for t in model.T:
            model.P_off_p[i,t] = variable(lb=0, value=0)
    model.P_off_n = variable_dict()
    for i in model.N:
        for t in model.T:
            model.P_off_n[i,t] = variable(lb=0, value=0)
    
    model.P_act = variable_dict()
    for i in model.N:
        for t in model.T:
            model.P_act[i,t] = variable(value=0)
    model.P_act_up = variable_dict()
    for i in model.N:
        for t in model.T:
            model.P_act_up[i,t] = variable(lb=0, value=0)
    model.P_act_down = variable_dict()
    for i in model.N:
        for t in model.T:
            model.P_act_down[i,t] = variable(lb=0, value=0)
    model.P_curt = variable_dict()
    for i in model.N:
        for t in model.T:
            model.P_curt[i,t] = variable(lb=0, value=0)
    model.P_NS = variable_dict()
    for i in model.N:
        for t in model.T:
            model.P_NS[i,t] = variable(lb=0, value=0)

    # Participation factors
    model.alpha = variable_dict()
    for i in model.N:
        for t in model.T:
            model.alpha[i,t] = variable(value=0)
    model.alpha_act = variable_dict()
    for i in model.N:
        for t in model.T:
            model.alpha_act[i,t] = variable(value=0)
    model.alpha_curt = variable_dict()
    for i in model.N:
        for t in model.T:
            model.alpha_curt[i,t] = variable(value=0, lb=0)
    model.alpha_NS = variable_dict()
    for i in model.N:
        for t in model.T:
            model.alpha_NS[i,t] = variable(value=0, lb=0)
    
    # Auxiliary variables for chance-constraints reformulation
    model.k_P = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.k_P[i,j,t] = variable(value=0)
    model.k_Q = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.k_Q[i,j,t] = variable(value=0)
    model.S_max = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.S_max[i,j,t] = variable(value=0, lb=0)
    
    # Auxiliary variables to help reformulate the SOC constraints
    model.x1 = variable_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                model.x1[i,w,t] = variable(value=0)
    model.x2 = variable_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                model.x2[i,w,t] = variable(value=0)
    model.x3 = variable_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                model.x3[i,w,t] = variable(value=0)
    model.x4 = variable_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                model.x4[i,w,t] = variable(value=0)
    model.x5 = variable_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                model.x5[i,j,w,t] = variable(value=0)
    model.x6 = variable_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                model.x6[i,j,w,t] = variable(value=0)
    model.x9 = variable_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                model.x9[i,j,w,t] = variable(value=0)
    model.x7 = variable_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                model.x7[i,j,w,t] = variable(value=0)
    model.x8 = variable_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                model.x8[i,j,w,t] = variable(value=0)
    model.x10 = variable_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                model.x10[i,j,w,t] = variable(value=0)
    model.x11 = variable_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                model.x11[i,w,t] = variable(value=0)
    model.x12 = variable_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                model.x12[i,w,t] = variable(value=0)
    
    model.r1 = variable_dict()
    for i in model.N:
        for t in model.T:
            model.r1[i,t] = variable(value=0, lb=0)
    model.r2 = variable_dict()
    for i in model.N:
        for t in model.T:
            model.r2[i,t] = variable(value=0, lb=0)
    model.r3 = variable_dict()
    for i in model.N:
        for t in model.T:
            model.r3[i,t] = variable(value=0, lb=0)
    model.r4 = variable_dict()
    for i in model.N:
        for t in model.T:
            model.r4[i,t] = variable(value=0, lb=0)
    model.r5 = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.r5[i,j,t] = variable(value=0, lb=0)
    model.r6 = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.r6[i,j,t] = variable(value=0, lb=0)
    model.r7 = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.r7[i,j,t] = variable(value=0, lb=0)
    model.r8 = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.r8[i,j,t] = variable(value=0, lb=0)
    model.r9 = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.r9[i,j,t] = variable(value=0, lb=0)
    model.r10 = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.r10[i,j,t] = variable(value=0, lb=0)
    model.r11 = variable_dict()
    for i in model.N:
        for t in model.T:
            model.r11[i,t] = variable(value=0, lb=0)
    model.r12 = variable_dict()
    for i in model.N:
        for t in model.T:
            model.r12[i,t] = variable(value=0, lb=0)
    
    #%% Objective function
    
    # model.min_costs = objective(baseMVA * (sum(model.P_off[o]*model.cost_off[o] for o in model.O) + sum(model.P_act_up[i, t]*model.cost_act_up[i] + model.P_act_down[i, t]*model.cost_act_down[i] + model.P_NS[i, t]*model.cost_NS[i] + model.P_curt[i, t]*model.cost_curt[i] for i in model.N for t in model.T)))
    
    # Without activation costs
    model.min_costs = objective(baseMVA * (sum(model.P_off[o]*model.cost_off[o] for o in model.O) + sum(model.P_NS[i, t]*model.cost_NS[i] + model.P_curt[i, t]*model.cost_curt[i] for i in model.N for t in model.T)))
    
    # # With DSO costs
    # obj = sum(model.P_act_up[i, t]*model.cost_act_up[i] + model.P_act_down[i, t]*model.cost_act_down[i] + model.P_NS[i, t]*model.cost_NS[i] + model.P_curt[i, t]*model.cost_curt[i] for i in model.N for t in model.T)
    # for o in model.O:
    #     i = offers.loc[o,'Bus']
    #     if offers.loc[o,'Direction']=='Up': 
    #         obj+=model.P_off[o]*(model.cost_off[o]-model.cost_req_up[i])
    #     elif offers.loc[o,'Direction']=='Down': 
    #         obj+=model.P_off[o]*(model.cost_off[o]-model.cost_req_down[i])
    # model.min_costs = objective(baseMVA * obj)
    
    #%% Constraints
    
    # Slack bus
    model.P_act_ref = constraint_dict()
    for t in model.T:
        model.P_act_ref[t] = constraint(body=model.P_act[ref,t],rhs=0)
    
    model.P_curt_ref = constraint_dict()
    for t in model.T:
        model.P_curt_ref[t] = constraint(body=model.P_curt[ref,t],rhs=0)
    
    model.P_NS_ref = constraint_dict()
    for t in model.T:
        model.P_NS_ref[t] = constraint(body=model.P_NS[ref,t],rhs=0)
    
    # Flows
    model.active_power_flow = constraint_dict()
    for k in model.N:
        for t in model.T:
            Lhs = sum(model.P[j, i, t] for j, i in model.B if i == k) - sum(model.P[i, j, t] for i, j in model.B if i == k) +\
                model.P_init[k, t] + model.P_wind[k, t] + model.P_act[k, t] - model.P_curt[k,t] + model.P_NS[k,t]
            if k == ref:
                Lhs += (- sum(model.P_act[i, t] - model.P_curt[i,t] + model.P_NS[i,t] for i in model.N))
            model.active_power_flow[k,t] = constraint(body=Lhs,rhs=0)
    
    model.reactive_power_flow = constraint_dict()
    for k in model.N:
        for t in model.T:
            Lhs = sum(model.Q[j, i, t] for j, i in model.B if i == k) - sum(model.Q[i, j, t] for i, j in model.B if i == k) +\
                factor_Q * (model.P_init[k, t] + model.P_wind[k, t] + model.P_act[k, t] - model.P_curt[k,t] + model.P_NS[k,t])
            if k == ref:
                Lhs += (-factor_Q * sum(model.P_act[i, t] - model.P_curt[i,t] + model.P_NS[i,t] for i in model.N))
            model.reactive_power_flow[k,t] = constraint(body=Lhs,rhs=0)
    
    model.voltage_drop = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            Lhs = model.V_sq[j, t]
            Rhs = model.V_sq[i, t] - 2 * (model.R[i, j] * model.P[i, j, t] + model.X[i, j] * model.Q[i, j, t])
            model.voltage_drop[i,j,t] = constraint(body=Lhs-Rhs,rhs=0)
    
    # Link between offers and nodes
    model.offers_up = constraint_dict()
    for i in model.N:
        for t in model.T:
            sum_offers_up = 0
            for o in model.O:
                if (offers.loc[o,'Bus']==i) and (offers.loc[o,'Direction']=='Up') and (offers.loc[o,'Time period']==1):
                    sum_offers_up+=model.P_off[o]
            model.offers_up[i,t] = constraint(body=model.P_off_p[i,t]-sum_offers_up, rhs=0)
    
    model.offers_down = constraint_dict()
    for i in model.N:
        for t in model.T:
            sum_offers_down = 0
            for o in model.O:
                if (offers.loc[o,'Bus']==i) and (offers.loc[o,'Direction']=='Down') and (offers.loc[o,'Time period']==1):
                    sum_offers_down+=model.P_off[o]
            model.offers_down[i,t] = constraint(body=model.P_off_n[i,t]-sum_offers_down, rhs=0)
    
    # Calculate x for the SOC constraints
    model.x1_def = constraint_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                Lhs = model.x1[i,w,t]
                Rhs = sum(model.alpha_act[i,t] * model.cov_mtx_sq[k,w] for k in model.W)
                model.x1_def[i,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.x2_def = constraint_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                Lhs = model.x2[i,w,t]
                Rhs = sum(model.alpha_act[i,t] * model.cov_mtx_sq[k,w] for k in model.W)
                model.x2_def[i,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.x3_def = constraint_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                Lhs = model.x3[i,w,t]
                Rhs = sum(sum(model.p_inc_mtx[m, n, i] * (model.R[m, n] + factor_Q * model.X[m, n])\
                                              * (model.xi_inc_mtx[j, k] - model.alpha[j,t]) * model.p_inc_mtx[m, n, j]\
                                                  for m, n in model.B for j in model.N) * model.cov_mtx_sq[k,w] for k in model.W)
                model.x3_def[i,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.x4_def = constraint_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                Lhs = model.x4[i,w,t]
                Rhs = sum(sum(model.p_inc_mtx[m, n, i] * (model.R[m, n] + factor_Q * model.X[m, n])\
                                              * (model.xi_inc_mtx[j, k] - model.alpha[j,t]) * model.p_inc_mtx[m, n, j]\
                                                  for m, n in model.B for j in model.N) * model.cov_mtx_sq[k,w] for k in model.W)
                model.x4_def[i,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.x5_def = constraint_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                Lhs = model.x5[i,j,w,t]
                Rhs = sum(sum(model.p_inc_mtx[i, j, m] * (model.xi_inc_mtx[m, k] - model.alpha[m,t]) for m in model.N)\
                                          * model.cov_mtx_sq[k,w] for k in model.W)
                model.x5_def[i,j,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.x6_def = constraint_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                Lhs = model.x6[i,j,w,t]
                Rhs = sum(sum(model.p_inc_mtx[i, j, m] * (model.xi_inc_mtx[m, k] - model.alpha[m,t]) for m in model.N)\
                                          * model.cov_mtx_sq[k,w] for k in model.W)
                model.x6_def[i,j,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.x9_def = constraint_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                Lhs = model.x9[i,j,w,t]
                Rhs = sum(sum(model.p_inc_mtx[i, j, m] * (model.xi_inc_mtx[m, k] - model.alpha[m,t]) for m in model.N)\
                                          * model.cov_mtx_sq[k,w] for k in model.W)
                model.x9_def[i,j,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.x7_def = constraint_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                Lhs = model.x7[i,j,w,t]
                Rhs = sum(factor_Q * sum(model.p_inc_mtx[i, j, m] * (model.xi_inc_mtx[m, k] - model.alpha[m,t]) for m in model.N)\
                                          * model.cov_mtx_sq[k,w] for k in model.W)
                model.x7_def[i,j,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.x8_def = constraint_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                Lhs = model.x8[i,j,w,t]
                Rhs = sum(factor_Q * sum(model.p_inc_mtx[i, j, m] * (model.xi_inc_mtx[m, k] - model.alpha[m,t]) for m in model.N)\
                                          * model.cov_mtx_sq[k,w] for k in model.W)
                model.x8_def[i,j,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.x10_def = constraint_dict()
    for i,j in model.B:
        for w in model.W:
            for t in model.T:
                Lhs = model.x10[i,j,w,t]
                Rhs = sum(factor_Q * sum(model.p_inc_mtx[i, j, m] * (model.xi_inc_mtx[m, k] - model.alpha[m,t]) for m in model.N)\
                                          * model.cov_mtx_sq[k,w] for k in model.W)
                model.x10_def[i,j,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.x11_def = constraint_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                Lhs = model.x11[i,w,t]
                Rhs = sum(model.alpha_curt[i,t] * model.cov_mtx_sq[k,w] for k in model.W)
                model.x11_def[i,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.x12_def = constraint_dict()
    for i in model.N:
        for w in model.W:
            for t in model.T:
                Lhs = model.x12[i,w,t]
                Rhs = sum(model.alpha_NS[i,t] * model.cov_mtx_sq[k,w] for k in model.W)
                model.x12_def[i,w,t] = constraint(body=Lhs-Rhs,rhs=0)
    
    # Calculate r for the SOC constraints
    model.r1_def = constraint_dict()
    for i in model.N:
        for t in model.T:
            Lhs = model.r1[i, t]
            Rhs = (model.P_act[i, t] + model.P_off_n[i, t]) / norm.ppf(1-A_viol_prob)
            model.r1_def[i,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.r2_def = constraint_dict()
    for i in model.N:
        for t in model.T:
            Lhs = model.r2[i, t]
            Rhs = (model.P_off_p[i, t] - model.P_act[i, t]) / norm.ppf(1-A_viol_prob)
            model.r2_def[i,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.r3_def = constraint_dict()
    for i in model.N:
        for t in model.T:
            Lhs = model.r3[i, t]
            Rhs = (model.V_sq[i, t] - model.Vmin_sq[i]) / norm.ppf(1 - V_viol_prob)
            model.r3_def[i,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.r4_def = constraint_dict()
    for i in model.N:
        for t in model.T:
            Lhs = model.r4[i, t]
            Rhs = (model.Vmax_sq[i] - model.V_sq[i, t]) / norm.ppf(1 - V_viol_prob)
            model.r4_def[i,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.r5_def = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            Lhs = model.r5[i, j, t]
            Rhs = (model.k_P[i, j, t] - model.P[i, j, t]) / norm.ppf(1-beta_factor*S_viol_prob*(1/1.25))
            model.r5_def[i, j, t] = constraint(body=Lhs-Rhs,rhs=0)
    model.r6_def = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            Lhs = model.r6[i, j, t]
            Rhs = (model.k_P[i, j, t] + model.P[i, j, t]) / norm.ppf(1-beta_factor*S_viol_prob*(1/1.25))
            model.r6_def[i, j, t] = constraint(body=Lhs-Rhs,rhs=0)
    model.r7_def = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            Lhs = model.r7[i, j, t]
            Rhs = (model.k_Q[i, j, t] - model.Q[i, j, t]) / norm.ppf(1 - (1-beta_factor) * S_viol_prob * (1 / 1.25))
            model.r7_def[i, j, t] = constraint(body=Lhs-Rhs,rhs=0)
    model.r8_def = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            Lhs = model.r8[i, j, t]
            Rhs = (model.k_Q[i, j, t] + model.Q[i, j, t]) / norm.ppf(1 - (1-beta_factor) * S_viol_prob * (1 / 1.25))
            model.r8_def[i, j, t] = constraint(body=Lhs-Rhs,rhs=0)
    model.r9_def = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            Lhs = model.r9[i, j, t]
            Rhs = model.k_P[i, j, t] / norm.ppf(1 - beta_factor*S_viol_prob*(1/2.5))
            model.r9_def[i, j, t] = constraint(body=Lhs-Rhs,rhs=0)
    model.r10_def = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            Lhs = model.r10[i, j, t]
            Rhs = model.k_Q[i, j, t] / norm.ppf(1 - (1-beta_factor) * S_viol_prob * (1 / 2.5))
            model.r10_def[i, j, t] = constraint(body=Lhs-Rhs,rhs=0)
    model.r11_def = constraint_dict()
    for i in model.N:
        for t in model.T:
            Lhs = model.r11[i, t]
            Rhs = model.P_curt[i, t]/ norm.ppf(1-C_viol_prob)
            model.r11_def[i,t] = constraint(body=Lhs-Rhs,rhs=0)
    model.r12_def = constraint_dict()
    for i in model.N:
        for t in model.T:
            Lhs = model.r12[i, t]
            Rhs = model.P_NS[i, t]/ norm.ppf(1-NS_viol_prob)
            model.r12_def[i,t] = constraint(body=Lhs-Rhs,rhs=0)
    
    # SOC constraints
    model.cc1 = constraint_dict()
    for i in model.N:
        for t in model.T:
            x = [model.x1[i,w,t] for w in model.W]
            model.cc1[i,t] = conic.quadratic(model.r1[i,t], x)
    model.cc2 = constraint_dict()
    for i in model.N:
        for t in model.T:
            x = [model.x2[i,w,t] for w in model.W]
            model.cc2[i,t] = conic.quadratic(model.r2[i,t], x)
    model.cc3 = constraint_dict()
    for i in model.N:
        for t in model.T:
            x = [model.x3[i,w,t] for w in model.W]
            model.cc3[i,t] = conic.quadratic(model.r3[i,t], x)
    model.cc4 = constraint_dict()
    for i in model.N:
        for t in model.T:
            x = [model.x4[i,w,t] for w in model.W]
            model.cc4[i,t] = conic.quadratic(model.r4[i,t], x)
    model.cc5 = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            x = [model.x5[i,j,w,t] for w in model.W]
            model.cc5[i,j,t] = conic.quadratic(model.r5[i,j,t], x)
    model.cc6 = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            x = [model.x6[i,j,w,t] for w in model.W]
            model.cc6[i,j,t] = conic.quadratic(model.r6[i,j,t], x)
    model.cc7 = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            x = [model.x7[i,j,w,t] for w in model.W]
            model.cc7[i,j,t] = conic.quadratic(model.r7[i,j,t], x)
    model.cc8 = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            x = [model.x8[i,j,w,t] for w in model.W]
            model.cc8[i,j,t] = conic.quadratic(model.r8[i,j,t], x)
    model.cc9 = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            x = [model.x9[i,j,w,t] for w in model.W]
            model.cc9[i,j,t] = conic.quadratic(model.r9[i,j,t], x)
    model.cc10 = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            x = [model.x10[i,j,w,t] for w in model.W]
            model.cc10[i,j,t] = conic.quadratic(model.r10[i,j,t], x)
    model.cc11 = constraint_dict()
    for i in model.N:
        for t in model.T:
            x = [model.x11[i,w,t] for w in model.W]
            model.cc11[i,t] = conic.quadratic(model.r11[i,t], x)
    model.cc12 = constraint_dict()
    for i in model.N:
        for t in model.T:
            x = [model.x12[i,w,t] for w in model.W]
            model.cc12[i,t] = conic.quadratic(model.r12[i,t], x)
    
    # Contraint on auxiliary variables k
    model.Smax_def = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            model.Smax_def[i,j,t] = constraint(body = model.S_max[i,j,t], rhs = model.Smax[i,j])
    
    model.k_lim = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            x = [model.k_P[i,j,t],model.k_Q[i,j,t]]
            model.k_lim[i,j,t] = conic.quadratic(model.S_max[i,j,t], x)
    
    # Sum of alphas
    model.alpha_tot = constraint_dict()
    for i in model.N:
        for t in model.T:
            model.alpha_tot[i,t] = constraint(body=model.alpha[i,t]-model.alpha_act[i,t]+model.alpha_curt[i,t]+model.alpha_NS[i,t], rhs=0)
    
    model.alpha_sum = constraint_dict()
    for t in model.T:
        model.alpha_sum[t] = constraint(body=(sum(model.alpha[i, t] for i in model.N)-model.alpha[ref,t]), rhs=0)
    
    # Activation
    model.P_act_def = constraint_dict()
    for i in model.N:
        for t in model.T:
            model.P_act_def[i,t]= constraint(body=model.P_act[i,t]-(model.P_act_up[i,t]-model.P_act_down[i,t]), rhs=0)

    #%% Specify solver settings and solve model
    solver = SolverFactory('mosek')
    
    if display_results == True:
        solver.solve(model, tee=True)
    else:
        solver.solve(model)
    
    P_act_res = np.array([(model.P_act[i, t].value)*baseMVA for i in model.N for t in model.T])
    P_act_up_res = np.array([(model.P_act_up[i, t].value)*baseMVA for i in model.N for t in model.T])
    P_act_down_res = np.array([(model.P_act_down[i, t].value)*baseMVA for i in model.N for t in model.T])
    P_curt_res = np.array([(model.P_curt[i, t].value)*baseMVA for i in model.N for t in model.T])
    P_NS_res = np.array([(model.P_NS[i, t].value)*baseMVA for i in model.N for t in model.T])
    P_off_p_res = np.array([(model.P_off_p[i, t].value)*baseMVA for i in model.N for t in model.T])
    P_off_n_res = np.array([(model.P_off_n[i, t].value)*baseMVA for i in model.N for t in model.T])
    V_sq_res = np.array([model.V_sq[i, t].value for i in model.N for t in model.T])
    alpha_res = np.array([model.alpha[i, t].value for i in model.N for t in model.T])
    alpha_act_res = np.array([model.alpha_act[i, t].value for i in model.N for t in model.T])
    alpha_curt_res = np.array([model.alpha_curt[i, t].value for i in model.N for t in model.T])
    alpha_NS_res = np.array([model.alpha_NS[i, t].value for i in model.N for t in model.T])
    line_loading = np.array([(model.P[i, j, t].value ** 2 + model.Q[i, j, t].value ** 2) /
    value(model.Smax[i, j] ** 2) for i,j in model.B for t in model.T])
    F_setpoints = {}
    F_setpoints['P_act'] = P_act_res
    F_setpoints['P_act_up'] = P_act_up_res
    F_setpoints['P_act_down'] = P_act_down_res
    F_setpoints['P_curt'] = P_curt_res
    F_setpoints['P_off_p'] = P_off_p_res
    F_setpoints['P_off_n'] = P_off_n_res
    F_setpoints['P_NS'] = P_NS_res
    F_setpoints['alpha'] = alpha_res
    F_setpoints['alpha_act'] = alpha_act_res
    F_setpoints['alpha_curt'] = alpha_curt_res
    F_setpoints['alpha_NS'] = alpha_NS_res
    F_setpoints['V_sq'] = V_sq_res
    F_setpoints['P'] = np.array([(model.P[i, j, t].value)*baseMVA for i,j in model.B for t in model.T])
    F_setpoints['costs'] = baseMVA*sum(model.P_off[o].value*model.cost_off[o].value for o in model.O)
    F_setpoints['obj'] = model.min_costs.expr()
    
    # Export accepted bids
    Accepted_df = pd.DataFrame(columns=['Type','Bus','Direction','Quantity','Price','Time_target','Market_Price'])
    for o in model.O:
        Poff = (model.P_off[o].value)*baseMVA
        if Poff >= 0.00001:
            market_price = model.dual[model.active_power_flow[offers.loc[o,'Bus'],offers.loc[o,'Time period']]]
            new_row = {"Type":'Offer', "Bus":offers.loc[o,'Bus'], "Direction":offers.loc[o,'Direction'], "Quantity":Poff, "Price":offers.loc[o,'Price'], "Time_target":offers.loc[o,'Time period'], "Market_Price":market_price}
            Accepted_df = Accepted_df.append(pd.Series(new_row, name=o), ignore_index=False)
            Accepted_df.index.name = "ID"
    Accepted_df.to_csv('{} Data/Results/{}_{}_accepted_SC.csv'.format(case_name,case_name, offers_file))
    
    # Export data per bus
    file_name = '{} Data/Results/results_bids_{}_{}.csv'.format(case_name, case_name, offers_file)
    bids_df = pd.read_csv(file_name, index_col=0)
    for i in model.N:
        if i != ref:
            Poff_up = (model.P_off_p[i, 1].value)*baseMVA
            if Poff_up < 0.00001:
                Poff_up == 0
            Poff_down = (model.P_off_n[i, 1].value)*baseMVA
            if Poff_down < 0.00001:
                Poff_down == 0
            bids_df.at[i, 'Accepted Offer Up - Stochastic'] = round(Poff_up,5)
            bids_df.at[i, 'Accepted Offer Down - Stochastic'] = round(Poff_down,5)
    bids_df.to_csv(file_name)
    
   # Export costs results 
    file_name = '{} Data/Results/{}_results_procurement.csv'.format(case_name,case_name)
    costs_df = pd.read_csv(file_name, index_col=None)
    SW = 0
    dso_costs = 0
    for o in model.O:
        i = offers.loc[o,'Bus']
        if offers.loc[o,'Direction']=='Up':
            cost_req = model.cost_req_up[i].value
        elif offers.loc[o,'Direction']=='Down':
            cost_req = model.cost_req_down[i].value
        SW += model.P_off[o].value* baseMVA* (cost_req - model.cost_off[o].value)
        dso_costs += model.P_off[o].value* baseMVA* cost_req
    if offers_file == 'high_liq':
        liq = 'High'
    elif offers_file == 'med_liq':
        liq = 'Medium'
    elif offers_file == 'low_liq':
        liq = 'Low'
    elif offers_file == 'no_liq':
        liq = 'No Offers'
    else:
        liq = 'Other'
    new_row={'Social Welfare': SW, 'DSO Costs': dso_costs, 'Liquidity':liq, 'Test Case':'Stochastic Market Clearing'}
    costs_df = costs_df.append(pd.Series(new_row), ignore_index=True)
    costs_df.to_csv(file_name, index=False)
    
    if display_results == True:
        print('Infeasible constraints check')
        print(log_infeasible_constraints(model, log_expression=True, log_variables=True))
        print('objective value ', model.min_costs.expr())
        print('alpha ', [round(p,3) for p in alpha_res])
        print('Resulting V ', [round(p,3) for p in V_sq_res])
        print('Line loading ', [round(100*p,1) for p in line_loading])
    
    return F_setpoints