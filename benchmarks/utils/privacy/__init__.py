from pfl.privacy import (
    NoPrivacy, 
    GaussianMechanism, 
    PLDPrivacyAccountant,
    #BandedMatrixFactorizationMechanism,
    LaplaceMechanism,
    NormClippingOnly,
    CentrallyApplicablePrivacyMechanism,
    CentrallyAppliedPrivacyMechanism
)


def parse_mechanism(mechanism_name,
                    clipping_bound=None,
                    epsilon=None,
                    delta=None,
                    order=None,
                    cohort_size=None,
                    noise_cohort_size=None,
                    num_epochs=None,
                    population=None,
                    min_separation=None,
                    is_central=False):
    """ Adjusted from https://github.com/apple/pfl-research"""
    if mechanism_name == 'none':
        mechanism = NoPrivacy()

    elif mechanism_name == 'gaussian':
        assert clipping_bound is not None
        assert epsilon is not None and delta is not None
        mechanism = GaussianMechanism.construct_single_iteration(
            clipping_bound, epsilon, delta)

    elif mechanism_name == 'gaussian_moments_accountant':
        assert clipping_bound is not None
        assert epsilon is not None
        assert delta is not None
        assert cohort_size is not None
        assert num_epochs is not None
        assert population is not None
        if noise_cohort_size is not None:
            noise_scale = cohort_size / noise_cohort_size
            max_cohort_size = max(cohort_size, noise_cohort_size)
        else:
            noise_scale = 1.0
            max_cohort_size = cohort_size
        accountant = PLDPrivacyAccountant(
            num_compositions=num_epochs,
            sampling_probability=max_cohort_size / population,
            mechanism='gaussian',
            epsilon=epsilon,
            delta=delta,
            noise_scale=noise_scale)
        mechanism = GaussianMechanism.from_privacy_accountant(
            accountant=accountant, clipping_bound=clipping_bound)
        
        #print(mechanism)
        #print(mechanism._relative_noise_stddev)

    #elif mechanism_name == 'banded_matrix_factorization':
    #    assert clipping_bound is not None
    #    assert epsilon is not None
    #    assert delta is not None
    #    assert cohort_size is not None
    #    assert num_epochs is not None
    #    assert population is not None
    #    assert min_separation is not None
    #    if noise_cohort_size is not None:
    #        noise_scale = cohort_size / noise_cohort_size
    #        max_cohort_size = max(cohort_size, noise_cohort_size)
    #    else:
    #        noise_scale = 1.0
    #        max_cohort_size = cohort_size

    #    make_privacy_accountant = lambda num_compositions: PLDPrivacyAccountant(
    #        num_compositions,
    #        sampling_probability=max_cohort_size / population,
    #        mechanism='gaussian',
    #        epsilon=epsilon,
    #        delta=delta,
    #        noise_scale=noise_scale)
    #    mechanism = BandedMatrixFactorizationMechanism(
    #        clipping_bound, num_epochs, min_separation,
    #        make_privacy_accountant)

    elif mechanism_name == 'laplace':
        assert clipping_bound is not None
        assert epsilon is not None
        mechanism = LaplaceMechanism(clipping_bound, epsilon)

    elif mechanism_name == 'norm_clipping_only':
        assert clipping_bound is not None
        assert order is not None
        mechanism = NormClippingOnly(order, clipping_bound)

    else:
        raise ValueError(
            "Please specify `mechanism_name`. If you don't want to use any privacy, specify 'none'."
        )

    if is_central:
        assert isinstance(mechanism, CentrallyApplicablePrivacyMechanism), (
            '`is_central=True` will wrap the mechanism into a central '
            f'mechanism, but {mechanism} is not centrally applicable')
        mechanism = CentrallyAppliedPrivacyMechanism(mechanism)

    return mechanism