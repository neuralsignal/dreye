"""Test Signal class
"""

import numpy as np
import os
from pytest import raises

from .context import dreye, constants, err, test_datapath


class TestSignal:

    def test_init(self):
        arr1 = np.random.random((100, 5))
        arr2 = np.arange(500).reshape(100, 5)
        domain = dreye.Domain(np.linspace(0, 10, 100), units='s')
        darr = np.linspace(0, 10, 100)

        self.s1 = dreye.Signals(
            arr1, domain, domain_units='ms',
            units='V'
        )
        self.s2 = dreye.Signals(
            arr2, domain, domain_units='s',
            units='V'
        )
        self.s3 = dreye.Signals(
            arr1, darr, domain_units='ms',
            units='mV'
        )
        self.s4 = dreye.Signals(
            arr2, domain, labels=['a', 'b', 'c', 'd', 'e'],
            domain_units='ms',
            units='km'
        )

    def test_add(self):
        self.test_init()

    def test_mul(self):
        self.test_init()

    def test_equality(self):
        self.test_init()

    def test_conversion(self):
        self.test_init()

    def test_attributes(self):
        self.test_init()

    def test_methods(self):
        self.test_init()

    def test_io(self):
        self.test_init()

    def test_plotting(self):
        self.test_init()

    def test_relplotting(self):
        self.test_init()

    def test_pca(self):
        self.test_init()
