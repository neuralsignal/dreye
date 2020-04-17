"""Test domain class
"""

import numpy as np
import os
from pytest import raises

from .context import dreye, constants, err, test_datapath


class TestDomain:

    def __init__(self):
        self.domain1 = dreye.Domain(-0.5, 1, 0.1, 's')
        self.domain2 = dreye.Domain(np.arange(-0.5, 1, 0.1), 's')
        self.domain3 = dreye.Domain([-0.5, 0, 0.5, 1.1], units='s')
        self.domain4 = dreye.Domain(-1.5, 0, 0.1, 's')
        self.domain5 = dreye.Domain(-0.5, 1, 0.1, 'V')
        self.domain6 = dreye.Domain(-0.5, 1, 0.2, 's')
        self.domain7 = dreye.Domain([0.1, 0.3, 0.4, 0.6, 0.8], units='s')
        self.domain_list = [
            self.domain1, self.domain2, self.domain3,
            self.domain4, self.domain5, self.domain6, self.domain7
        ]

    def test_init(self):
        assert isinstance(self.domain1, dreye.Domain)
        assert isinstance(self.domain2, dreye.Domain)
        with raises(err.DreyeError):
            # non-unique
            dreye.Domain([0.1, 0.1, 0.2, 0.2, 0.3, 0.3], units='s')

    def test_add(self):
        assert self.domain1 == (self.domain4 + 1)
        assert (self.domain1 + 10).end == 11.

    def test_mul(self):
        assert (self.domain1 * 3).start == -1.5
        assert (
            (self.domain1 * constants.ureg('s')).units
            == constants.ureg('s**2').units
        )

    def test_equality(self):
        assert self.domain1 == self.domain1.copy()
        assert self.domain1 == (self.domain4 + (1 * constants.ureg('s')))
        assert self.domain1 != self.domain2
        assert self.domain1 != self.domain3
        assert self.domain1 != self.domain4
        assert self.domain1 != self.domain5
        assert self.domain1 != self.domain5

    def test_conversion(self):
        assert self.domain1.to('ms') != self.domain1
        assert self.domain1.to('ms').units == constants.ureg('ms').units

    def test_attributes(self):
        assert self.domain1.is_uniform
        assert not self.domain3.is_uniform
        assert self.domain1.start == -0.5
        assert self.domain1.end == 1
        assert self.domain.interval == 0.1
        assert self.domain1.span == 1.5
        assert isinstance(self.domain3.interval, np.ndarray)
        assert isinstance(self.domain1.units, constants.ureg.Unit)
        assert isinstance(self.domain1.magnitude, np.ndarray)
        assert isinstance(self.domain1.gradient, constants.ureg.Quantity)
        assert isinstance(self.domain1.values, constants.ureg.Quantity)

    def test_methods(self):
        with raises(err.DreyeError):
            self.domain1.equalize_domains(self.domain7)
        with raises(err.DreyeUnitError):
            self.domain1.equalize_domains(self.domain5)

        new_domain = self.domain1.equalize_domains(
            self.domain7.enforce_uniformity())

        with raises(err.DreyeError):
            self.domain7.start = 0
        with raises(err.DreyeUnitError):
            self.domain1.units = 'V'

        self.domain1.start = 0

        assert new_domain.start == 0.1
        assert new_domain.end == 0.8
        assert np.round(new_domain.interval, 3) == 0.175

    def test_io(self):
        filepath = os.path.join(test_datapath, 'domain_test.json')
        for _domain in self.domain_list:
            _domain.save(filepath)
            domain = dreye.Domain.load(filepath)
            assert domain == _domain
