"""
Linear DistFlow AC power flow according to the Baran and Wu formulaiton.
Chnace-constraints for bus voltage and line flow power included.

09.06.2021 ID
"""
import numpy as np
from pyomo.kernel import *
from pyomo.util.infeasible import log_infeasible_constraints
import pandas as pd

def cc_lin_dist_flow_model(network_data, accepted_offers, wind_scn, pf, costs, display_results, check_line_loading, case_name, test_case, market_name, wind_scn_id):
    
    #Extract data
    baseMVA = network_data['base_MVA']
    node_data = network_data['bus_data']
    branch_data = network_data['branch_data']
    wf_data = network_data['wf_data']
    set_point = np.array(network_data['setpoint'])
    
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

    # Sets
    model.T = range(1,len_T+1)
    model.N = node_data['Bus'] # nodes
    model.B = [(branch_data.loc[i, 'From'], branch_data.loc[i, 'To'])
                                            for i in branch_data.index]  # lines
    
    # Nodes with windfarm and wind
    wind_bus = []
    wind_id = []
    wind_val = []
    id_wind = 0
    for b in node_data['Bus'].index:
        if node_data['Bus'][b] in wf_data['Bus'].values:
            wind_bus.append(node_data['Bus'][b])
            wind_id.append(b)
            wind_val.append(wind_scn[id_wind])
            id_wind+=1
        else:
            wind_val.append(0)
    model.W = wind_bus
    
    #%% Parameters
    
    # Costs
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
    
    # Nodes
    model.Vmin_sq = parameter_dict()
    for i in node_data.index:
        model.Vmin_sq[node_data.loc[i, 'Bus']] = parameter(node_data.loc[i, 'Vmin'] ** 2)
    model.Vmax_sq = parameter_dict()
    for i in node_data.index:
        model.Vmax_sq[node_data.loc[i, 'Bus']] = parameter(node_data.loc[i, 'Vmax'] ** 2)

    # Setpoint
    factor_Q = np.sqrt((1-pf ** 2)/pf ** 2)
    P_ref = -(sum(set_point)+sum(wind_val))
    
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
            model.P_wind[node_data.loc[i, 'Bus'], t] = parameter(wind_val[i] /baseMVA)
    model.Q_wind = parameter_dict()
    for i in node_data.index:
        for t in model.T:
            model.Q_wind[node_data.loc[i, 'Bus'], t] = parameter(factor_Q * wind_val[i] /baseMVA)

    # Lines
    model.Smax_sq = parameter_dict()
    for i in branch_data.index:
        model.Smax_sq[branch_data.loc[i, 'From'], branch_data.loc[i, 'To']] = parameter(branch_data.loc[i, 'Lim']**2 /baseMVA)
    model.R = parameter_dict()
    for i in branch_data.index:
        model.R[branch_data.loc[i, 'From'], branch_data.loc[i, 'To']] = parameter(branch_data.loc[i, 'R'])
    model.X = parameter_dict()
    for i in branch_data.index:
        model.X[branch_data.loc[i, 'From'], branch_data.loc[i, 'To']] = parameter(branch_data.loc[i, 'X'])
    
    # Limits activation
    model.act_max = parameter_dict()
    for i in node_data.index:
        for t in model.T:
            max_act = 0
            for o in accepted_offers.index:
                if accepted_offers.loc[o,'Bus']==node_data.loc[i, 'Bus'] and accepted_offers.loc[o,'Direction']=='Up' and accepted_offers.loc[o,'Time_target']==t:
                    max_act += accepted_offers.loc[o,'Quantity']
            model.act_max[node_data.loc[i, 'Bus'], t] = parameter(max_act)
    
    model.act_min = parameter_dict()
    for i in node_data.index:
        for t in model.T:
            min_act = 0
            for o in accepted_offers.index:
                if accepted_offers.loc[o,'Bus']==node_data.loc[i, 'Bus'] and accepted_offers.loc[o,'Direction']=='Down' and accepted_offers.loc[o,'Time_target']==t:
                    min_act += accepted_offers.loc[o,'Quantity']
            model.act_min[node_data.loc[i, 'Bus'], t] = parameter(-min_act)

    #%% Variables
    
    model.V_sq = variable_dict()
    for i in model.N:
        for t in model.T:
            model.V_sq[i,t] = variable(value=1, lb=model.Vmin_sq[i], ub=model.Vmax_sq[i])
    model.P = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.P[i,j,t] = variable(value=0)
    model.Q = variable_dict()
    for i,j in model.B:
        for t in model.T:
            model.Q[i,j,t] = variable(value=0)
    model.P_act = variable_dict()
    for i in model.N:
        for t in model.T:
            model.P_act[i,t] = variable(value=0, lb=model.act_min[i,t], ub=model.act_max[i,t])
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
    
    #%% Objective function
    
    model.min_costs = objective(sum(model.P_act_up[i, t]*model.cost_act_up[i] + model.P_act_down[i, t]*model.cost_act_down[i] + model.P_NS[i, t]*model.cost_NS[i] + model.P_curt[i, t]*model.cost_curt[i] for i in model.N for t in model.T))
    # model.min_costs = objective(1)
    
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
                Lhs += (-sum(model.P_act[i, t] - model.P_curt[i,t] + model.P_NS[i,t] for i in model.N))
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
    
    model.S_lim = constraint_dict()
    for i,j in model.B:
        for t in model.T:
            model.S_lim[i,j,t] = constraint(body=model.P[i,j,t]**2 + model.Q[i,j,t]**2,ub=model.Smax_sq[i,j])
    
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
    V_sq_res = np.array([(model.V_sq[i, t].value)*baseMVA for i in model.N for t in model.T])
    line_loading = np.array([(model.P[i, j, t].value ** 2 + model.Q[i, j, t].value ** 2) /
    value(model.Smax_sq[i, j]) for i,j in model.B for t in model.T])
    line_flows = np.array([baseMVA*sqrt(model.P[i, j, t].value ** 2 + model.Q[i, j, t].value ** 2) for i,j in model.B for t in model.T])
    costs_RT = baseMVA*sum(model.P_act_up[i, t].value*model.cost_act_up[i].value + model.P_act_down[i, t].value*model.cost_act_down[i].value + model.P_NS[i, t].value*model.cost_NS[i].value + model.P_curt[i, t].value*model.cost_curt[i].value for i in model.N for t in model.T)
    F_setpoints = {}
    F_setpoints['P_act'] = P_act_res
    F_setpoints['P_act_up'] = P_act_up_res
    F_setpoints['P_act_down'] = P_act_down_res
    F_setpoints['P_curt'] = P_curt_res
    F_setpoints['P_NS'] = P_NS_res
    F_setpoints['V_sq'] = V_sq_res
    F_setpoints['P'] = np.array([(model.P[i, j, t].value)*baseMVA for i,j in model.B for t in model.T])
    F_setpoints['Costs'] = costs_RT
    
    if display_results == True:
        print('Infeasible constraints check')
        print(log_infeasible_constraints(model, log_expression=True, log_variables=True))
        print('objective value ', model.min_costs.expr())
        print('Resulting V ', [round(p,3) for p in V_sq_res])
        print('Line loading ', [round(100*p,1) for p in line_loading])
        print('Line flows ', [round(p,3) for p in line_flows])
        
    if check_line_loading == True:
        file_name = '{}_line_loading.csv'.format(case_name)
        lines_df = pd.read_csv(file_name, index_col=None)
        new_row = {}
        for i,j in model.B:
            l = line_id[i,j]
            new_row['Line {}'.format(l)] = baseMVA*sqrt(model.P[i, j, 1].value ** 2 + model.Q[i, j, 1].value ** 2)
        lines_df = lines_df.append(pd.Series(new_row), ignore_index=True)
        lines_df.to_csv(file_name, index=False)
    
    # Export costs results 
    file_name = '{}_results_RT.csv'.format(case_name)
    costs_df = pd.read_csv(file_name, index_col=None)
    if test_case == 'high_liq':
        liq = 'High'
    elif test_case == 'med_liq':
        liq = 'Medium'
    elif test_case == 'low_liq':
        liq = 'Low'
    elif test_case == 'no_liq':
        liq = 'No Offers'
    if market_name == 'SC':
        market = 'Stochastic Market Clearing'
    elif market_name == 'DC_no_zones':
        market = 'No Zones'
    elif market_name == 'DC_conservative_zones':
        market = 'Conservative Zones'
    elif market_name == 'DC_exact_zones':
        market = 'Exact Zones'
    new_row={'DSO Costs': costs_RT, 'Liquidity':liq, 'Test Case':market}
    costs_df = costs_df.append(pd.Series(new_row), ignore_index=True)
    costs_df.to_csv(file_name, index=False)
    
    #Export curtailment results
    file_name = '{}_{}_{}_curtailment.csv'.format(case_name, test_case, market_name)
    curt_df = pd.read_csv(file_name, index_col=0)
    epsilon = 0.0000001
    t=1
    for i in model.N:
        Pcurt = (model.P_curt[i, 1].value)*baseMVA
        PNS = (model.P_NS[i, 1].value)*baseMVA
        if Pcurt > epsilon and PNS <= epsilon:
            curt_df.at[wind_scn_id,'LS {}'.format(i)] = - Pcurt
        elif PNS > epsilon and Pcurt <= epsilon:
            curt_df.at[wind_scn_id,'LS {}'.format(i)] = PNS
        elif PNS <= epsilon and Pcurt <= epsilon:
            curt_df.at[wind_scn_id,'LS {}'.format(i)] = 0
        elif PNS > epsilon and Pcurt > epsilon:
            print('error LS')
        Pact = (model.P_act[i, t].value)*baseMVA
        if Pact > epsilon:
            curt_df.at[wind_scn_id,'Act {}'.format(i)] = Pact
        else:
            curt_df.at[wind_scn_id,'Act {}'.format(i)] = 0
    curt_df.to_csv(file_name)
    
    return F_setpoints