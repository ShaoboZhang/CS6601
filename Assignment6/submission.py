import numpy as np
import Assignment6.hmm_submission_test as tests


def gaussian_prob(x, para_tuple):
    """
    Compute the pdf of a given x value and (mean, std)
    """
    if para_tuple == (None, None):
        return 0.0
    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std ** 2) ** -0.5 * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    return gaussian_percentile


def part_1_a():
    """
    Provide probabilities for the word HMMs outlined below.
    """
    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.,
        'B3': 0.,
        'Bend': 0.,
    }
    b_transition_probs = {
        'B1': {'B1': 0.625, 'B2': 0.375, 'B3': 0., 'Bend': 0.},
        'B2': {'B1': 0., 'B2': 0.625, 'B3': 0.375, 'Bend': 0.},
        'B3': {'B1': 0., 'B2': 0., 'B3': 0.625, 'Bend': 0.375},
        'Bend': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 1.000},
    }
    # Parameters for end state is not required
    b_emission_paras = {
        'B1': (41.750, 2.773),
        'B2': (58.625, 5.678),
        'B3': (53.125, 5.418),
        'Bend': (None, None)
    }

    """Word CAR"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.,
        'C3': 0.,
        'Cend': 0.,
    }
    c_transition_probs = {
        'C1': {'C1': 0.667, 'C2': 0.333, 'C3': 0., 'Cend': 0.},
        'C2': {'C1': 0., 'C2': 0.000, 'C3': 1.000, 'Cend': 0.},
        'C3': {'C1': 0., 'C2': 0., 'C3': 0.800, 'Cend': 0.200},
        'Cend': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 1.000},
    }
    # Parameters for end state is not required
    c_emission_paras = {
        'C1': (35.667, 4.899),
        'C2': (43.667, 1.700),
        'C3': (44.200, 7.341),
        'Cend': (None, None)
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.,
        'H3': 0.,
        'Hend': 0.,
    }
    # Probability of a state changing to another state.
    h_transition_probs = {
        'H1': {'H1': 0.667, 'H2': 0.333, 'H3': 0., 'Hend': 0.},
        'H2': {'H1': 0., 'H2': 0.857, 'H3': 0.143, 'Hend': 0.},
        'H3': {'H1': 0., 'H2': 0., 'H3': 0.812, 'Hend': 0.188},
        'Hend': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 1.00},
    }
    # Parameters for end state is not required
    h_emission_paras = {
        'H1': (45.333, 3.972),
        'H2': (34.952, 8.127),
        'H3': (67.438, 5.733),
        'Hend': (None, None)
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def viterbi(evidence_vector, states, prior_probs, transition_probs, emission_paras):
    """
    Viterbi Algorithm to calculate the most likely states give the evidence.
    """
    if not evidence_vector:
        return [], 0.0
    seq = [[s] for s in states]
    probs = []
    for s in states:
        if s not in ['B1', 'C1', 'H1']:
            probs.append(0)
        else:
            probs.append(gaussian_prob(evidence_vector[0], emission_paras[s]) * prior_probs[s])
    for evid in evidence_vector[1:]:
        temp_seq = seq.copy()
        temp_prob = probs.copy()
        for i, cur_s in enumerate(states):
            cur_prob = []
            for j, pre_s in enumerate(states):
                if cur_s not in transition_probs[pre_s].keys():
                    cur_prob.append(0)
                elif cur_s in ['Bend', 'Cend', 'Hend']:
                    cur_prob.append(0)
                else:
                    p = probs[j] * transition_probs[pre_s][cur_s] * gaussian_prob(evid, emission_paras[cur_s])
                    cur_prob.append(p)
            temp_prob[i] = np.max(cur_prob)
            idx = int(np.argmax(cur_prob))
            temp_seq[i] = seq[idx] + [cur_s]
        seq = temp_seq.copy()
        probs = temp_prob.copy()
    idx = int(np.argmax(probs))
    return seq[idx], probs[idx]


def part_2_a():
    """
    Provide probabilities for the word HMMs outlined below.
    """
    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.,
        'B3': 0.,
        'Bend': 0.,
    }
    # example: {'B1': {'B1' : (right-hand Y, left-hand Y), ... }
    b_transition_probs = {
        'B1': {'B1': (0.625, 0.700), 'B2': (0.375, 0.300), 'B3': (0., 0.), 'Bend': (0., 0.)},
        'B2': {'B1': (0., 0.), 'B2': (0.625, 0.050), 'B3': (0.375, 0.950), 'Bend': (0., 0.)},
        # 'B3': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0.625, 0.727), 'Bend': (0.375, 0.273)},
        'B3': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0.625, 0.727), 'Bend': (0.125, 0.091), 'C1': (0.125, 0.091),
               'H1': (0.125, 0.091)},
        'Bend': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0., 0.), 'Bend': (1.000, 1.000)},
    }
    # example: {'B1': [(right-mean, right-std), (left-mean, left-std)] ...}
    b_emission_paras = {
        'B1': [(41.750, 2.773), (108.200, 17.314)],
        'B2': [(58.625, 5.678), (78.670, 1.886)],
        'B3': [(53.125, 5.418), (64.182, 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    """Word Car"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.,
        'C3': 0.,
        'Cend': 0.,
    }
    c_transition_probs = {
        'C1': {'C1': (0.667, 0.700), 'C2': (0.333, 0.300), 'C3': (0., 0.), 'Cend': (0., 0.)},
        'C2': {'C1': (0., 0.), 'C2': (0.000, 0.625), 'C3': (1.000, 0.375), 'Cend': (0., 0.)},
        # 'C3': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0.800, 0.625), 'Cend': (0.200, 0.375)},
        'C3': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0.800, 0.625), 'Cend': (0.067, 0.125), 'B1': (0.067, 0.125),
               'H1': (0.067, 0.125)},
        'Cend': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0., 0.), 'Cend': (1.000, 1.000)},
    }
    c_emission_paras = {
        'C1': [(35.667, 4.899), (56.300, 10.659)],
        'C2': [(43.667, 1.700), (37.110, 4.306)],
        'C3': [(44.200, 7.341), (50.000, 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.,
        'H3': 0.,
        'Hend': 0.,
    }
    h_transition_probs = {
        'H1': {'H1': (0.667, 0.700), 'H2': (0.333, 0.300), 'H3': (0., 0.), 'Hend': (0., 0.)},
        'H2': {'H1': (0., 0.), 'H2': (0.857, 0.842), 'H3': (0.143, 0.158), 'Hend': (0., 0.)},
        # 'H3': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0.812, 0.824), 'Hend': (0.188, 0.176)},
        'H3': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0.812, 0.824), 'Hend': (0.063, 0.059), 'B1': (0.063, 0.059),
               'C1': (0.063, 0.059)},
        'Hend': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0., 0.), 'Hend': (1.000, 1.000)},
    }
    h_emission_paras = {
        'H1': [(45.333, 3.972), (53.600, 7.392)],
        'H2': [(34.952, 8.127), (37.168, 8.875)],
        'H3': [(67.438, 5.733), (74.176, 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def inner(x):
    return x[0] * x[1]


def multidimensional_viterbi(evidence_vector, states, prior_probs, transition_probs, emission_paras):
    """
    Decode the most likely word phrases generated by the evidence vector.
    States, prior_probs, transition_probs, and emission_probs will now contain all the words from part_2_a.
    """
    if not evidence_vector:
        return [], 0.0
    seq = [[s] for s in states]
    probs = []
    for s in states:
        if s not in ['B1', 'C1', 'H1']:
            probs.append(0)
        else:
            p = gaussian_prob(evidence_vector[0][0], emission_paras[s][0]) * \
                gaussian_prob(evidence_vector[0][1], emission_paras[s][1]) * prior_probs[s]
            probs.append(p)
    for evid in evidence_vector[1:]:
        temp_prob = probs.copy()
        temp_seq = seq.copy()
        for i, cur_s in enumerate(states):
            cur_prob = []
            for j, pre_s in enumerate(states):
                if cur_s not in transition_probs[pre_s].keys():
                    cur_prob.append(0)
                elif cur_s in ['Bend', 'Cend', 'Hend']:
                    cur_prob.append(0)
                else:
                    p = gaussian_prob(evid[0], emission_paras[cur_s][0]) * \
                        gaussian_prob(evid[1], emission_paras[cur_s][1])
                    p *= probs[j] * inner(transition_probs[pre_s][cur_s])
                    cur_prob.append(p)
            temp_prob[i] = np.max(cur_prob)
            idx = int(np.argmax(cur_prob))
            temp_seq[i] = seq[idx] + [cur_s]
        probs = temp_prob.copy()
        seq = temp_seq.copy()
    idx = int(np.argmax(probs))
    return seq[idx], probs[idx]


def return_your_name():
    return 'Xingbo Song'


if __name__ == '__main__':
    tests.TestPart1b().test_viterbi_realsample1(part_1_a, viterbi)
    tests.TestPart1b().test_viterbi_realsample2(part_1_a, viterbi)
    tests.TestPart1b().test_viterbi_realsample3(part_1_a, viterbi)
    pass
