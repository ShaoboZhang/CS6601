from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import random, numpy as np
from scipy.stats import norm


def make_power_plant_net():
    BayesNet = BayesianModel()
    BayesNet.add_node('temperature')
    BayesNet.add_node('faulty gauge')
    BayesNet.add_node('gauge')
    BayesNet.add_node('faulty alarm')
    BayesNet.add_node('alarm')
    BayesNet.add_edge('temperature', 'faulty gauge')
    BayesNet.add_edge('temperature', 'gauge')
    BayesNet.add_edge('faulty gauge', 'gauge')
    BayesNet.add_edge('gauge', 'alarm')
    BayesNet.add_edge('faulty alarm', 'alarm')
    return BayesNet


def set_probability(bayes_net):
    # 0.8 normal for false, 0.2 high for true
    cpd_t = TabularCPD('temperature', 2, values=[[0.8], [0.2]])
    cpd_fg = TabularCPD('faulty gauge', 2, values=[[0.95, 0.2], [0.05, 0.8]], evidence=['temperature'],
                        evidence_card=[2])
    cpd_g = TabularCPD('gauge', 2, values=[[0.95, 0.05, 0.2, 0.8], [0.05, 0.95, 0.8, 0.2]],
                       evidence=['faulty gauge', 'temperature'], evidence_card=[2, 2])
    cpd_fa = TabularCPD('faulty alarm', 2, values=[[0.85], [0.15]])
    cpd_a = TabularCPD('alarm', 2, values=[[0.9, 0.1, 0.55, 0.45], [0.1, 0.9, 0.45, 0.55]],
                       evidence=['faulty alarm', 'gauge'], evidence_card=[2, 2])
    bayes_net.add_cpds(cpd_a, cpd_fa, cpd_g, cpd_fg, cpd_t)
    return bayes_net


def get_alarm_prob(bayes_net):
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    prob = marginal_prob['alarm'].values
    return prob[1]


def get_gauge_prob(bayes_net):
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    prob = marginal_prob['gauge'].values
    return prob[1]


def get_temperature_prob(bayes_net):
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'],
                                    evidence={'alarm': 1, 'faulty alarm': 0, 'faulty gauge': 0}, joint=False)
    prob = conditional_prob['temperature'].values
    return prob[1]


def get_game_network():
    BayesNet = BayesianModel()
    BayesNet.add_node('A')
    BayesNet.add_node('B')
    BayesNet.add_node('C')
    BayesNet.add_node('AvB')
    BayesNet.add_node('BvC')
    BayesNet.add_node('CvA')
    BayesNet.add_edge('A', 'AvB')
    BayesNet.add_edge('B', 'AvB')
    BayesNet.add_edge('B', 'BvC')
    BayesNet.add_edge('C', 'BvC')
    BayesNet.add_edge('C', 'CvA')
    BayesNet.add_edge('A', 'CvA')
    cpd_a = TabularCPD('A', 4, values=[[0.15], [0.45], [0.3], [0.1]])
    cpd_b = TabularCPD('B', 4, values=[[0.15], [0.45], [0.3], [0.1]])
    cpd_c = TabularCPD('C', 4, values=[[0.15], [0.45], [0.3], [0.1]])
    cpd_avb = TabularCPD('AvB', 3,
                         values=[[0.10, 0.20, 0.15, 0.05,
                                  0.60, 0.10, 0.20, 0.15,
                                  0.75, 0.60, 0.10, 0.20,
                                  0.90, 0.75, 0.60, 0.10],
                                 [0.10, 0.60, 0.75, 0.90,
                                  0.20, 0.10, 0.60, 0.75,
                                  0.15, 0.20, 0.10, 0.60,
                                  0.05, 0.15, 0.20, 0.10],
                                 [0.80, 0.20, 0.10, 0.05,
                                  0.20, 0.80, 0.20, 0.10,
                                  0.10, 0.20, 0.80, 0.20,
                                  0.05, 0.10, 0.20, 0.80]],
                         evidence=['A', 'B'],
                         evidence_card=[4, 4])
    cpd_bvc = TabularCPD('BvC', 3,
                         values=[[0.10, 0.20, 0.15, 0.05,
                                  0.60, 0.10, 0.20, 0.15,
                                  0.75, 0.60, 0.10, 0.20,
                                  0.90, 0.75, 0.60, 0.10],
                                 [0.10, 0.60, 0.75, 0.90,
                                  0.20, 0.10, 0.60, 0.75,
                                  0.15, 0.20, 0.10, 0.60,
                                  0.05, 0.15, 0.20, 0.10],
                                 [0.80, 0.20, 0.10, 0.05,
                                  0.20, 0.80, 0.20, 0.10,
                                  0.10, 0.20, 0.80, 0.20,
                                  0.05, 0.10, 0.20, 0.80]],
                         evidence=['B', 'C'],
                         evidence_card=[4, 4])
    cpd_cva = TabularCPD('CvA', 3,
                         values=[[0.10, 0.20, 0.15, 0.05,
                                  0.60, 0.10, 0.20, 0.15,
                                  0.75, 0.60, 0.10, 0.20,
                                  0.90, 0.75, 0.60, 0.10],
                                 [0.10, 0.60, 0.75, 0.90,
                                  0.20, 0.10, 0.60, 0.75,
                                  0.15, 0.20, 0.10, 0.60,
                                  0.05, 0.15, 0.20, 0.10],
                                 [0.80, 0.20, 0.10, 0.05,
                                  0.20, 0.80, 0.20, 0.10,
                                  0.10, 0.20, 0.80, 0.20,
                                  0.05, 0.10, 0.20, 0.80]],
                         evidence=['C', 'A'],
                         evidence_card=[4, 4])
    BayesNet.add_cpds(cpd_a, cpd_b, cpd_c, cpd_avb, cpd_bvc, cpd_cva)
    return BayesNet


def calculate_posterior(bayes_net):
    """calculate P(BvC|AvB=0,CvA=2)"""
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'], evidence={'AvB': 0, 'CvA': 2}, joint=False, show_progress=False)
    posterior = conditional_prob['BvC'].values
    return posterior  # list


def Gibbs_sampler(bayes_net, initial_state):
    """
    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    """
    if not initial_state:
        sample = list(np.random.randint(0, 4, size=[3, ])) + [0, random.randint(0, 2), 2]
        return tuple(sample)

    sample = list(initial_state)
    ran_idx = random.randint(0, 3)
    if ran_idx < 3:
        if ran_idx == 0:
            weights = [0.05212617, 0.39779233, 0.43337813, 0.11670337]
        elif ran_idx == 1:
            weights = [0.31203638, 0.44550888, 0.19055735, 0.05189739]
        else:
            weights = [0.07383111, 0.46460579, 0.37960479, 0.08195831]
        new_val = random.choices([0, 1, 2, 3], weights=weights)
        sample[ran_idx] = new_val[0]
    else:
        weights = [0.25890074, 0.42796763, 0.31313163]
        new_val = random.choices([0, 1, 2], weights=weights)
        sample[4] = new_val[0]

    return tuple(sample)


# def MH_sampler(bayes_net, initial_state):
#     """
#     initial_state is a list of length 6 where:
#     index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
#     index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
#     """
#     if not initial_state:
#         sample = list(np.random.randint(0, 4, size=[3, ])) + [0, random.randint(0, 2), 2]
#         return tuple(sample)
#
#     A_cpd = bayes_net.get_cpds('A')
#     AvB_cpd = bayes_net.get_cpds("AvB")
#     team_table = A_cpd.values
#     match_table = AvB_cpd.values
#
#     sample = list(initial_state)
#     nodes = list(bayes_net.nodes())
#     val_dict = {node: val for val, node in zip(sample, nodes)}
#
#     for idx, val in enumerate(sample):
#         if idx < 3:
#             weights = [norm.pdf(i, sample[idx], 1) for i in range(4)]
#             new_val = random.choices([0, 1, 2, 3], weights=weights)[0]
#             prob1 = team_table[new_val]
#             children = bayes_net.get_children(nodes[idx])
#             for child in children:
#                 parents = bayes_net.get_parents(child)
#                 prob1 = prob1 * match_table[val_dict[child], val_dict[parents[0]], val_dict[parents[1]]]
#             prob0 = team_table[val]
#             children = bayes_net.get_children(nodes[idx])
#             for child in children:
#                 parents = bayes_net.get_parents(child)
#                 prob0 = prob0 * match_table[val_dict[child], val_dict[parents[0]], val_dict[parents[1]]]
#         elif idx == 4:
#             weights = [norm.pdf(i, sample[idx], 1) for i in range(3)]
#             new_val = random.choices([0, 1, 2], weights=weights)[0]
#             prob1 = match_table[val, val_dict['B'], val_dict['C']]
#             prob0 = match_table[new_val, val_dict['B'], val_dict['C']]
#         else:
#             continue
#         alpha = min(1.0, prob1 / prob0)
#         if random.uniform(0, 1) < alpha:
#             sample[idx] = new_val
#
#     return tuple(sample)

def MH_sampler(bayes_net, initial_state):
    """
    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    """
    if not initial_state:
        sample = list(np.random.randint(0, 4, size=[3, ])) + [0, random.randint(0, 2), 2]
        return tuple(sample)


    sample = list(initial_state)
    nodes = list(bayes_net.nodes())
    val_dict = {node: val for val, node in zip(sample, nodes)}

    solver = VariableElimination(bayes_net)

    for idx, val in enumerate(sample):
        evid = {node: val for node, val in zip(nodes, sample) if node != nodes[idx]}
        if idx < 3:
            weights = [norm.pdf(i, sample[idx], 1) for i in range(4)]
            new_val = random.choices([0, 1, 2, 3], weights=weights)[0]
            prob = solver.query(variables=[nodes[idx]], evidence={'AvB': 0, 'CvA':2}, joint=False, show_progress=False)
            # prob = solver.query(variables=[nodes[idx]], evidence=evid, joint=False, show_progress=False)
            prob = prob[nodes[idx]].values
            prob1 = prob[new_val]
            prob0 = prob[val]
        elif idx == 4:
            weights = [norm.pdf(i, sample[idx], 1) for i in range(3)]
            new_val = random.choices([0, 1, 2], weights=weights)[0]
            prob = solver.query(variables=[nodes[idx]], evidence={'AvB': 0, 'CvA': 2}, joint=False, show_progress=False)
            # prob = solver.query(variables=['BvC'], evidence=evid, joint=False, show_progress=False)
            prob = prob['BvC'].values
            prob1 = prob[new_val]
            prob0 = prob[val]
        else:
            continue
        alpha = min(1.0, prob1 / prob0)
        if random.uniform(0, 1) < alpha:
            sample[idx] = new_val

    return tuple(sample)


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""

    Gibbs_count = 0
    Gibbs_convergence = np.array([0, 0, 0])
    MH_count = 0
    MH_rejection_count = 0
    MH_convergence = np.array([0, 0, 0])

    delta = 0.001

    # gibbs sample
    n = 0
    N = 50
    sample = initial_state if initial_state else Gibbs_sampler(bayes_net, None)
    bvc_count = np.array([0, 0, 0])
    while n < N:
        Gibbs_count += 1
        bvc_count[sample[4]] += 1
        diff = max(np.abs(Gibbs_convergence - bvc_count / Gibbs_count))
        if diff < delta:
            n += 1
        else:
            n = 0
        Gibbs_convergence = bvc_count / Gibbs_count
        sample = Gibbs_sampler(bayes_net, sample)
    print(Gibbs_count, Gibbs_convergence)

    # mh sample
    n = 0
    # N = 50
    sample = initial_state if initial_state else MH_sampler(bayes_net, None)
    bvc_count = np.array([0, 0, 0])
    while n < N:
        MH_count += 1
        bvc_count[sample[4]] += 1
        diff = max(np.abs(MH_convergence - bvc_count / MH_count))
        if diff < delta:
            n += 1
        else:
            MH_rejection_count += 1
            n = 0
        MH_convergence = bvc_count / MH_count
        sample = MH_sampler(bayes_net, sample)

    print(MH_count)
    print(MH_convergence)

    print(MH_count / Gibbs_count)
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    choice = 0
    options = ['Gibbs', 'Metropolis-Hastings']
    factor = 1
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    return 'Xingbo Song'


if __name__ == '__main__':
    net = get_game_network()

    # solver = VariableElimination(net)
    # conditional_prob = solver.query(variables=['BvC'], evidence={'AvB': 0, 'CvA':2}, joint=False, show_progress=False)
    # posterior = conditional_prob['BvC'].values
    # print(posterior)

    # check for 2e
    print(calculate_posterior(net))
    compare_sampling(net, None)
