'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Verify all-date static references and resource-comparison boundaries.
Synthetic candidate records test analysis semantics, not physical feasibility.
'''
import copy
import unittest

from scripts.report_evidence import seasonal_comparison


class SeasonalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.domains = [{'scenario': 1, 'date': day} for day in ('a', 'b')]
        rows = [{'tuple': [1, 0, 0], 'mass': 2., 'cost': 100., 'outcome': 'success'},
                {'tuple': [2, 0, 0], 'mass': 3., 'cost': 80., 'outcome': 'success'},
                {'tuple': [1, 0, 1], 'mass': 1., 'cost': 50., 'outcome': 'failed'}]
        self.candidates = {f's1_{day}': copy.deepcopy(rows) for day in ('a', 'b')}
        self.selected = [{'scenario': 1, 'date': day, 'mass_kg': 2., 'cost_usd': 100.,
                          'battery_units': 1, 'pv_units': 0, 'floor_deficit_wh': 0.,
                          'pv_used_energy_wh': 0., 'electrical_load_energy_wh': 10.}
                         for day in ('a', 'b')]

    def test_objectives_choose_different_full_season_references(self):
        result = seasonal_comparison(self.domains, self.candidates, self.selected)[0]
        self.assertEqual(result['common_accepted_count'], 2)
        self.assertEqual(result['references']['mass_first']['tuple'], [1, 0, 0])
        cost = result['references']['cost_first']
        self.assertEqual(cost['tuple'], [2, 0, 0])
        self.assertEqual(cost['saving_summary']['cost_saving_percent']['mean'], -25.)
        self.assertIsNone(result['pv_limit_count'])

    def test_one_failed_or_unresolved_date_excludes_tuple(self):
        for outcome in ('failed', 'unresolved'):
            with self.subTest(outcome=outcome):
                self.candidates['s1_b'][0]['outcome'] = outcome
                result = seasonal_comparison(self.domains, self.candidates, self.selected)[0]
                self.assertEqual(result['common_accepted_count'], 1)
                self.assertEqual(result['references']['mass_first']['tuple'], [2, 0, 0])

    def test_empty_day_cannot_be_dropped_from_season(self):
        for row in self.candidates['s1_b']:
            row['outcome'] = 'failed'
        result = seasonal_comparison(self.domains, self.candidates, self.selected[:1])[0]
        self.assertEqual(result['dates'], ['a', 'b'])
        self.assertEqual(result['common_accepted_count'], 0)
        self.assertEqual(result['references'], {'mass_first': None, 'cost_first': None})

    def test_missing_date_does_not_silently_become_full_coverage(self):
        del self.candidates['s1_b']
        with self.assertRaises(KeyError):
            seasonal_comparison(self.domains, self.candidates, self.selected)

    def test_changed_resource_catalogue_is_not_a_fixed_design(self):
        self.candidates['s1_b'][0]['mass'] = 4.
        with self.assertRaisesRegex(ValueError, 'resource catalogue varies'):
            seasonal_comparison(self.domains, self.candidates, self.selected)


if __name__ == '__main__':
    unittest.main()
