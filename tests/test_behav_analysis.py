import pytest
import warnings
import torch
import numpy as np
import src.behav_analysis as ba
import constants.constants_expt1 as constants_expt_1
import types

import os


class TestWrapAngle:

    def test_edge_case(self):
        """ Test the edge case - pi should be converted to -pi."""
        assert ba.wrap_angle(np.pi) == -np.pi

    def test_typical(self):
        assert ba.wrap_angle(2 * np.pi) == 0
        assert ba.wrap_angle(3 * np.pi) == -np.pi


class TestGetAbsErrMeanSd:
    def test_input_multi_dim(self):
        """ Test the case where input is a multidimensional array. Should be internally flattened prior to
        calculating the output."""
        input_array = torch.tensor([[torch.pi, torch.pi, torch.pi], [0, 0, 0]])
        result1, result2 = ba.get_abs_err_mean_sd(input_array)
        assert result1 == np.degrees(torch.pi / 2), 'Multidimensional array inputs: Mean not calculated correctly'
        assert result2 == 90, 'Multidimensional array inputs: SD not calculated correctly'

    def test_warning(self):
        with pytest.warns(Warning):
            warnings.warn(
                'Large angular values detected. Check that the input array contains angular errors in radians, '
                'if so - wrap the values to the [-pi, pi] interval. Converting the values to radians.', Warning)

    def test_input_degrees(self):
        """ Test the case where input is given in degrees. Suppresses the warning, input should be internally
        converted to radians prior to calculating the output."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            input_array = [180, 90, 120]
            result1, result2 = ba.get_abs_err_mean_sd(input_array)
            assert result1 == 130, 'Inputs in degrees case: Mean not calculated correctly.'
            assert result2 == pytest.approx(
                37.416573867739416), 'Inputs in degrees case: SD not calculated correctly.'

    def test_input_negative(self):
        input_array = [np.radians(180), np.radians(-90), np.radians(120)]
        result1, result2 = ba.get_abs_err_mean_sd(input_array)
        assert result1 == 130, 'Negative inputs case: Mean not calculated correctly.'
        assert result2 == pytest.approx(
            37.416573867739416), 'Negative inputs case: SD not calculated correctly.'


class TestGetAngularErrors:
    def test_no_errors(self):
        """
        Test a case where choices equal the probed values, i.e. there are no errors.
        """
        n_colours = 3
        choices = torch.linspace(0, np.pi, n_colours)
        probed_colours = torch.linspace(0, np.pi, n_colours)
        result = ba.get_angular_errors(choices, probed_colours)
        assert torch.all(result == torch.zeros((n_colours, 1))), 'No errors case does not output 0 error values.'

    def test_constant_output(self):
        """
        Test a case where there is only one value for all choices, errors should be the respective distances from the
        probed values.
        """
        n_colours = 3
        choices = torch.zeros((n_colours,))
        probed_colours = torch.linspace(0, np.pi, n_colours)
        result = ba.get_angular_errors(choices, probed_colours)
        result = result.squeeze()  # squeeze so that tensor is 1D, like probed_colours
        assert torch.all(result == -probed_colours), 'Constant output case does not output the correct error values.'


class TestLoadMixtureModelParams:
    def test_cue_validity_1(self):
        with pytest.raises(NotImplementedError, 'Mixture models only fitted for data from Experiment 4, cue validity < 1.'):
            ba.load_mixture_model_params(constants_expt_1)

    def test_files_exist(self):
        # not the most elegant solution - create a bogus constants module with the MATLAB file path set to the
        # current directory

        constants_test = types.ModuleType("temporary_module")
        constants_test.PARAMS = {'cue_validity': 0.75, 'MATLAB_PATH': os.getcwd(), 'n_models': 1}

        with pytest.raises(FileNotFoundError):
            ba.load_mixture_model_params(constants_test)

